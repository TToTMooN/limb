"""PolicyAgent — runs an external policy model via a PolicyClient.

Composes PolicyClient (transport) + ObsTransform + ActionTransform +
ActionChunkManager into the Agent protocol. Supports sync and async modes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from dm_env.specs import Array
from loguru import logger

from limb.agents.agent import PolicyAgent
from limb.agents.constants import ActionSpec
from limb.agents.policy_learning.action_chunk_manager import ActionChunkManager
from limb.agents.policy_learning.transforms import ActionTransform, ObsTransform
from limb.robots.utils import Rate
from limb.utils.portal_utils import remote


@dataclass
class YamPolicyAgent(PolicyAgent):
    """Agent that runs an external policy server via a PolicyClient.

    Parameters
    ----------
    client : PolicyClient
        Transport layer (OpenPIClient or WebSocketPolicyClient).
    obs_transform : ObsTransform
        Preprocesses limb observations for the server.
    action_transform : ActionTransform
        Converts flat action arrays to limb action dicts.
    action_horizon : int
        Number of actions per chunk (must match server).
    smoothing_window : int
        Overlap window for blending consecutive chunks.
    async_inference : bool
        If True, run inference in a background thread and serve from buffer.
        If False, inference blocks act().
    inference_interval_s : float | None
        For async mode: minimum seconds between inference calls.
    use_joint_state_as_action : bool
        If True, action includes both pos and vel per arm.
    """

    client: Any = None  # PolicyClient — Any for _target_ instantiation
    obs_transform: Any = field(default_factory=ObsTransform)
    action_transform: ActionTransform = field(default_factory=ActionTransform)
    action_horizon: int = 25
    smoothing_window: int = 4
    async_inference: bool = True
    inference_interval_s: Optional[float] = None
    use_joint_state_as_action: bool = False
    # SubRL: expose the RLT features to a wrapping SubtaskRLAgent — z_rl (pi0.5 image
    # embedding, needs the serve run with SUBRL_RETURN_EMBED=1) + the reference action
    # chunk. Added to act()'s output as "_z_rl" / "_ref_chunk". Off by default.
    expose_rl_features: bool = False

    def __post_init__(self) -> None:
        self._chunk_mgr = ActionChunkManager(
            action_horizon=self.action_horizon,
            smoothing_window=self.smoothing_window,
        )
        self._last_embed: Optional[np.ndarray] = None    # z_rl
        self._last_chunk: Optional[np.ndarray] = None     # ref_chunk (horizon, action_dim)
        self._obs_lock = threading.Lock()
        self._latest_obs: Optional[Dict[str, Any]] = None
        self._step_counter = 0
        self._starved = False                             # freeze-guard state (see act())
        self._last_flat_action: Optional[np.ndarray] = None
        # Bumped on every reset() so in-flight async inferences whose request
        # predates the reset can be detected and discarded instead of
        # repopulating the buffer with pre-takeover actions.
        self._generation = 0

        if self.async_inference:
            self._inference_rate = (
                Rate(1.0 / self.inference_interval_s, rate_name="policy_inference")
                if self.inference_interval_s
                else None
            )
            self._thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._thread.start()
            logger.info(
                "YamPolicyAgent started (async, interval={}s, horizon={}, smoothing={})",
                self.inference_interval_s,
                self.action_horizon,
                self.smoothing_window,
            )
        else:
            logger.info(
                "YamPolicyAgent started (sync, horizon={}, smoothing={})",
                self.action_horizon,
                self.smoothing_window,
            )

    def _inference_loop(self) -> None:
        """Background thread: continuously infer and update the chunk buffer."""
        while True:
            while self._latest_obs is None:
                time.sleep(0.05)

            with self._obs_lock:
                obs = self._latest_obs
                request_step = self._step_counter
                request_gen = self._generation

            try:
                result = self.client.infer(obs)
                actions = result["actions"]  # (horizon, action_dim)
                if self.expose_rl_features:
                    self._capture_features(result)
            except Exception as e:
                logger.warning("Inference failed: {}", e)
                time.sleep(0.1)
                continue

            # If reset() fired while we were inferring (DAgger takeover, etc.),
            # this chunk is based on a pre-takeover observation — drop it
            # rather than letting it leak into the buffer post-reset.  Hold
            # the lock across both the gen check AND the buffer update so
            # reset() can't slip between them and let a stale chunk repopulate
            # a just-cleared buffer.
            with self._obs_lock:
                if request_gen != self._generation:
                    logger.debug(
                        "Discarding stale inference (gen {} != {})",
                        request_gen,
                        self._generation,
                    )
                    continue
                elapsed_steps = self._step_counter - request_step
                self._chunk_mgr.update(actions, steps_since_request=elapsed_steps)

            if self._inference_rate is not None:
                self._inference_rate.sleep()

    @remote()
    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        transformed_obs = self.obs_transform(obs)

        if self.async_inference:
            with self._obs_lock:
                self._latest_obs = transformed_obs
                self._step_counter += 1

            # FREEZE GUARD (on-robot freeze 2026-07-08 18:08): this wait used to be
            # unbounded — when the serve's websocket died the buffer never refilled and
            # the whole control loop hung here forever (robot holding, TUI frozen, no
            # pedals). Bounded wait: HOLD the last commanded action while the inference
            # thread reconnects. First starvation waits 5 s; subsequent ticks 0.25 s so
            # the loop stays responsive through an outage.
            deadline = time.monotonic() + (0.25 if self._starved else 5.0)
            while not self._chunk_mgr.has_actions and time.monotonic() < deadline:
                time.sleep(0.01)

            if self._chunk_mgr.has_actions:
                if self._starved:
                    logger.info("Action chunks flowing again — resuming policy control")
                self._starved = False
                flat_action = self._chunk_mgr.get_action()
                self._last_flat_action = flat_action
            elif self._last_flat_action is not None:
                if not self._starved:
                    logger.error(
                        "No action chunk after 5 s (policy server down?) — HOLDING the "
                        "last commanded action until inference recovers")
                self._starved = True
                flat_action = self._last_flat_action
            else:
                # Startup only: nothing has been commanded yet, so there is no pose to
                # hold — the original unbounded wait is the safe behavior here.
                while not self._chunk_mgr.has_actions:
                    time.sleep(0.01)
                flat_action = self._chunk_mgr.get_action()
                self._last_flat_action = flat_action
        else:
            # Sync: block on inference, then buffer
            if not self._chunk_mgr.has_actions or self._chunk_mgr.remaining == 0:
                result = self.client.infer(transformed_obs)
                if self.expose_rl_features:
                    self._capture_features(result)
                self._chunk_mgr.update(result["actions"])
            flat_action = self._chunk_mgr.get_action()

        action = self.action_transform(flat_action)
        if self.expose_rl_features:
            action["_z_rl"] = self._last_embed                 # (2048,) or None
            action["_ref_chunk"] = self._last_chunk            # (horizon, action_dim) or None
            # Execution cursor into _ref_chunk: index of the action just returned. Lets the
            # RL side slice an ALIGNED reference window (review H10). remaining counts the
            # actions still buffered AFTER this one.
            try:
                action["_ref_cursor"] = max(0, self.action_horizon - self._chunk_mgr.remaining - 1)
            except Exception:
                action["_ref_cursor"] = 0
        return action

    def _capture_features(self, result: Dict[str, Any]) -> None:
        emb = result.get("image_embedding")
        if emb is None and not getattr(self, "_warned_no_embed", False):
            # review M22: silently-null z_rl poisons the replay for hours before anyone
            # notices — the serve must run with SUBRL_RETURN_EMBED=1.
            self._warned_no_embed = True
            logger.warning(
                "expose_rl_features=True but the policy server returned no 'image_embedding' "
                "(launch it with SUBRL_RETURN_EMBED=1) — z_rl will be None and RL transitions "
                "will be degraded")
        self._last_embed = None if emb is None else np.asarray(emb, np.float32)
        self._last_chunk = np.asarray(result["actions"], np.float32)

    @remote()
    def reset(self) -> None:
        """Clear any buffered action chunks and any pending observation.

        Called by composite agents (e.g. DAggerAgent) when handing control back
        to the policy after a takeover so stale chunks don't replay.  The
        background inference thread will populate a fresh chunk on the next
        observation.
        """
        # Clear the chunk buffer AND bump the generation atomically with the
        # inference loop's gen-check/update region.  This guarantees an
        # in-flight async inference is either fully rejected by the gen check
        # or fully applied before we wipe it — never half-applied after the
        # reset returns.
        with self._obs_lock:
            self._chunk_mgr.reset()
            self._latest_obs = None
            self._step_counter = 0
            self._generation += 1
            gen = self._generation
        logger.info("YamPolicyAgent reset (chunk buffer cleared, gen={})", gen)

    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        spec = {}
        for name in self.action_transform.arm_names:
            arm_spec: Dict[str, Array] = {
                "pos": Array(shape=(self.action_transform.joints_per_arm,), dtype=np.float32)
            }
            if self.action_transform.include_vel:
                arm_spec["vel"] = Array(shape=(self.action_transform.joints_per_arm,), dtype=np.float32)
            spec[name] = arm_spec
        return spec

    def close(self) -> None:
        self.client.close()
