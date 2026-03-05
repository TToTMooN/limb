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

    def __post_init__(self) -> None:
        self._chunk_mgr = ActionChunkManager(
            action_horizon=self.action_horizon,
            smoothing_window=self.smoothing_window,
        )
        self._obs_lock = threading.Lock()
        self._latest_obs: Optional[Dict[str, Any]] = None
        self._step_counter = 0

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

            try:
                result = self.client.infer(obs)
                actions = result["actions"]  # (horizon, action_dim)
            except Exception as e:
                logger.warning("Inference failed: {}", e)
                time.sleep(0.1)
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

            while not self._chunk_mgr.has_actions:
                time.sleep(0.01)

            flat_action = self._chunk_mgr.get_action()
        else:
            # Sync: block on inference, then buffer
            if not self._chunk_mgr.has_actions or self._chunk_mgr.remaining == 0:
                result = self.client.infer(transformed_obs)
                self._chunk_mgr.update(result["actions"])
            flat_action = self._chunk_mgr.get_action()

        return self.action_transform(flat_action)

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
