"""Episode replay as an Agent — composable with DAgger for human takeover.

Wraps a recorded episode (the same artifact ``limb replay`` consumes) into
the :class:`limb.agents.agent.Agent` protocol.  Each ``act()`` call returns
one step from the recording and advances an internal pointer; when the
recording runs out the agent either holds the final pose, loops back to
step 0, or returns an empty action (robots hold via PD).

The intended use is as :attr:`DAggerAgent.inner_policy`:

    AUTONOMOUS  -> followers retrace the recorded trajectory; leaders mirror.
    PAUSED      -> step pointer frozen; everything holds.
    CORRECTING  -> bilateral teleop drives followers via the leaders.  The
                   replay pointer stays frozen, so when AUTO resumes after
                   the correction the recording continues from where it
                   was paused (not rewound).

Standalone use (no DAgger) also works — just point ``agent`` at this class
in any teleop/record config to verify that an episode reproduces the
expected motion on hardware.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from dm_env.specs import Array
from loguru import logger

from limb.agents.agent import Agent
from limb.replay import _load_episode_data
from limb.utils.portal_utils import remote

_VALID_ON_END = {"hold", "loop", "stop"}


@dataclass
class EpisodeReplayAgent(Agent):
    """Replay a recorded episode tick-by-tick as an Agent.

    Parameters
    ----------
    episode_dir : str
        Path to the episode directory (e.g. ``recordings/.../episode_*``).
        Must contain ``timestamps.npy`` and per-arm ``*_actions.npz`` (or
        falls back to ``*_states.npz`` if actions aren't recorded).
    on_end : str
        Behavior when the recording is exhausted:
        ``"hold"`` (default) — keep emitting the final step's action so
        the robot stays where the recording ended.
        ``"loop"`` — restart from step 0.
        ``"stop"`` — emit ``{}`` so ``RobotEnv.step`` skips ``_apply_action``
        and the PD controllers hold whatever the last commanded pose was.
    arm_keys : Sequence[str] | None
        Restrict replay to these arms (keys must exist in the recording).
        ``None`` (default) replays every arm with action data present.
    use_joint_state_as_action : bool
        Required by :class:`Agent`; replay always commands position so this
        stays ``False``.
    """

    episode_dir: str = ""
    on_end: str = "hold"
    arm_keys: Optional[Sequence[str]] = None

    use_joint_state_as_action: bool = False

    _step: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.on_end not in _VALID_ON_END:
            raise ValueError(f"on_end must be one of {_VALID_ON_END}, got {self.on_end!r}")
        if not self.episode_dir:
            raise ValueError("EpisodeReplayAgent requires episode_dir")

        ep_path = Path(self.episode_dir).expanduser()
        if not ep_path.exists():
            raise FileNotFoundError(f"Episode directory not found: {ep_path}")

        data = _load_episode_data(ep_path)
        timestamps = data["timestamps"]
        self._n_steps = len(timestamps)
        if self._n_steps == 0:
            raise ValueError(f"Episode {ep_path} has no timesteps")

        # Prefer recorded actions (they include the gripper and exactly match
        # what was commanded).  Fall back to states for legacy episodes that
        # never wrote actions.
        action_arrays: Dict[str, np.ndarray] = {}
        for arm_name, arm_actions in data["actions"].items():
            if "pos" in arm_actions:
                action_arrays[arm_name] = np.asarray(arm_actions["pos"])
        if not action_arrays:
            logger.info("No *_actions.npz in {}; falling back to *_states.npz", ep_path)
            for arm_name, arm_states in data["arms"].items():
                if "joint_pos" not in arm_states:
                    continue
                jp = np.asarray(arm_states["joint_pos"])
                gp = arm_states.get("gripper_pos")
                if gp is not None and len(gp) == len(jp):
                    action_arrays[arm_name] = np.concatenate([jp, np.asarray(gp).reshape(-1, 1)], axis=1)
                else:
                    action_arrays[arm_name] = jp

        if not action_arrays:
            raise ValueError(f"Episode {ep_path} has no action or state data to replay")

        if self.arm_keys is not None:
            wanted = tuple(self.arm_keys)
            missing = [k for k in wanted if k not in action_arrays]
            if missing:
                raise KeyError(f"arm_keys {missing} not in episode {ep_path} (available: {list(action_arrays)})")
            action_arrays = {k: action_arrays[k] for k in wanted}

        # Sanity-check that every arm's action stream has the same length.
        lengths = {k: v.shape[0] for k, v in action_arrays.items()}
        if len(set(lengths.values())) != 1:
            logger.warning(
                "Arm action streams have unequal lengths: {} (will replay min length)",
                lengths,
            )
            min_len = min(lengths.values())
            action_arrays = {k: v[:min_len] for k, v in action_arrays.items()}
            self._n_steps = min(self._n_steps, min_len)

        self._actions: Dict[str, np.ndarray] = action_arrays
        self._arm_keys: tuple[str, ...] = tuple(action_arrays.keys())
        self._action_dim: Dict[str, int] = {k: v.shape[1] for k, v in action_arrays.items()}

        logger.info(
            "EpisodeReplayAgent ready: {} arms, {} steps, on_end={}, dir={}",
            len(self._arm_keys),
            self._n_steps,
            self.on_end,
            ep_path,
        )

    # ---------------------------------------------------------------- #
    #  Agent protocol                                                  #
    # ---------------------------------------------------------------- #

    @remote()
    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        del obs  # replay is open-loop
        with self._lock:
            if self._step >= self._n_steps:
                if self.on_end == "loop":
                    self._step = 0
                elif self.on_end == "hold":
                    # Repeat the final step indefinitely.
                    idx = self._n_steps - 1
                    return {arm: {"pos": self._actions[arm][idx].copy()} for arm in self._arm_keys}
                else:  # stop
                    return {}

            idx = self._step
            self._step += 1
            return {arm: {"pos": self._actions[arm][idx].copy()} for arm in self._arm_keys}

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        return {arm: {"pos": Array(shape=(self._action_dim[arm],), dtype=np.float32)} for arm in self._arm_keys}

    @remote()
    def reset(self) -> None:
        """No-op for replay.

        Composite agents call ``inner_policy.reset()`` to flush stale internal
        state (e.g. action-chunk buffers).  Replay is open-loop and has none,
        so reset is a no-op.  Use :meth:`rewind` to explicitly restart
        playback from step 0.
        """

    @remote()
    def rewind(self) -> None:
        """Reset playback to step 0."""
        with self._lock:
            self._step = 0
        logger.info("EpisodeReplayAgent rewound to step 0")

    @remote()
    def step_index(self) -> int:
        """Return the next step index that ``act()`` would emit."""
        with self._lock:
            return int(self._step)

    def close(self) -> None:
        logger.info("EpisodeReplayAgent closed")
