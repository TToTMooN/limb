"""
YAM-to-YAM bilateral teleoperation agent.

For setups with 4 YAM arms (2 leader + 2 follower), this agent reads the
joint positions of the leader arms (which are operator-backdriven, in
``zero_torque_mode``) from the observation dict, and commands the follower
arms to mirror them.

Wiring:

* All four arms are listed in the launch config's ``robots:`` dict so the
  env collects observations from each one.
* The leader arms are listed in the launch config's ``release_at_startup:``
  field so they are placed in ``zero_torque_mode`` before the control
  loop starts -- the operator can then move them freely.
* Followers stay under PID position control and receive commands derived
  from leader joint positions on every tick.

Differs from :class:`limb.agents.teleoperation.yam_gello_agent.YamGelloAgent`:
the leader is another YAM (read via Portal RPC + obs dict), not a Dynamixel
device.  Kinematics are identical, so joint limits and gripper indices match
between leader and follower with no mapping required.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np
from dm_env.specs import Array
from loguru import logger

from limb.agents.agent import Agent

# Set LIMB_GRIPPER_DEBUG=1 to append per-tick leader gripper readings and the
# computed follower gripper commands to this file (throttled).  Lets us see
# whether the leader gripper signal is changing and whether it reaches the
# follower, independent of the TUI which can clobber subprocess stderr.
_GRIPPER_DEBUG = bool(os.environ.get("LIMB_GRIPPER_DEBUG"))
_GRIPPER_DEBUG_PATH = "/tmp/limb_gripper_debug.log"
from limb.utils.portal_utils import remote

# YAM joint limits (radians) -- mirrors robot_configs/yam/left.yaml.
# Hardcoded here (rather than read from a robot config) so the agent
# stays self-contained at the Portal RPC boundary.
_YAM_JOINT_LIMITS = np.array(
    [
        [-2.09, 3.14],
        [0.00, 3.14],
        [0.05, 3.14],
        [-1.35, 1.35],
        [-1.50, 1.50],
        [-2.00, 2.00],
    ],
    dtype=np.float64,
)


@dataclass
class YamYamBilateralAgent(Agent):
    """Drive YAM follower arms from YAM leader arms read via the obs dict.

    Parameters
    ----------
    follower_keys : Sequence[str]
        Keys identifying follower arms in the obs/action dicts.
    leader_keys : Sequence[str]
        Keys identifying leader arms in the obs dict, in the same order
        as ``follower_keys``.
    joint_signs : Sequence[int]
        Per-joint sign correction applied to leader joint positions
        before sending to the follower.  Default ``(1, 1, 1, 1, 1, 1)``
        since 4 identical YAMs share calibration; override if the
        leaders are mounted in a mirrored or rotated frame.
    gripper_passthrough : bool
        When True (default), the follower's gripper command is the
        leader's gripper position.  When False, ``default_gripper_value``
        is sent on every tick (e.g. for testing arm motion only).
    default_gripper_value : float
        Gripper command used when ``gripper_passthrough`` is False or
        when a leader arm has no ``gripper_pos`` in its observation.
    """

    follower_keys: Sequence[str] = field(default_factory=lambda: ("left", "right"))
    leader_keys: Sequence[str] = field(default_factory=lambda: ("leader_left", "leader_right"))
    joint_signs: Sequence[int] = field(default_factory=lambda: (1, 1, 1, 1, 1, 1))
    gripper_passthrough: bool = True
    default_gripper_value: float = 0.0

    use_joint_state_as_action: bool = False

    def __post_init__(self) -> None:
        # Coerce OmegaConf-loaded sequences to plain tuples + numpy arrays
        self.follower_keys = tuple(self.follower_keys)
        self.leader_keys = tuple(self.leader_keys)
        if len(self.follower_keys) != len(self.leader_keys):
            raise ValueError(
                f"follower_keys ({self.follower_keys}) and leader_keys ({self.leader_keys}) must have the same length"
            )
        self._signs = np.asarray(tuple(self.joint_signs), dtype=np.float64)
        if self._signs.shape != (_YAM_JOINT_LIMITS.shape[0],):
            raise ValueError(f"joint_signs must have length {_YAM_JOINT_LIMITS.shape[0]}, got {self._signs.shape[0]}")
        logger.info(
            "YamYamBilateralAgent: leaders {} -> followers {} (gripper_passthrough={})",
            list(self.leader_keys),
            list(self.follower_keys),
            self.gripper_passthrough,
        )

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        action: Dict[str, Dict[str, np.ndarray]] = {}
        lo = _YAM_JOINT_LIMITS[:, 0]
        hi = _YAM_JOINT_LIMITS[:, 1]
        dbg_rows = []

        for fkey, lkey in zip(self.follower_keys, self.leader_keys, strict=True):
            leader_obs = obs.get(lkey)
            if leader_obs is None:
                raise KeyError(
                    f"YamYamBilateralAgent: obs missing leader entry '{lkey}'. "
                    f"Available keys: {[k for k in obs.keys() if not k.startswith('_')]}"
                )

            joints = np.asarray(leader_obs["joint_pos"], dtype=np.float64) * self._signs
            joints = np.clip(joints, lo, hi)

            if self.gripper_passthrough and "gripper_pos" in leader_obs:
                gripper = np.asarray(leader_obs["gripper_pos"], dtype=np.float64).reshape(-1)
                grip_src = "leader"
            else:
                gripper = np.array([self.default_gripper_value], dtype=np.float64)
                grip_src = "default"

            action[fkey] = {"pos": np.concatenate([joints, gripper])}

            if _GRIPPER_DEBUG:
                lg = leader_obs.get("gripper_pos")
                fobs = obs.get(fkey) or {}
                fg = fobs.get("gripper_pos")
                fmt = lambda v: None if v is None else round(float(np.asarray(v).reshape(-1)[0]), 4)
                dbg_rows.append(
                    f"{lkey}->{fkey} leader={fmt(lg)} cmd={round(float(gripper[0]),4)} "
                    f"follower_actual={fmt(fg)} src={grip_src}"
                )

        if _GRIPPER_DEBUG and dbg_rows:
            self._debug_write(dbg_rows)

        return action

    def _debug_write(self, rows) -> None:
        """Append one throttled (~2 Hz) line per act() with, for each pair:
        leader gripper reading, the follower gripper command, and the
        follower's *actual* gripper reading from obs.  If `cmd` tracks
        `leader` but `follower_actual` does not track `cmd`, the break is on
        the follower side (mode / force limiter / calibration), not the
        passthrough.  Only active when LIMB_GRIPPER_DEBUG is set.
        """
        now = time.monotonic()
        if now - getattr(self, "_dbg_last_t", 0.0) < 0.5:
            return
        self._dbg_last_t = now
        try:
            with open(_GRIPPER_DEBUG_PATH, "a") as f:
                f.write(f"{now:.2f}  " + "  |  ".join(rows) + "\n")
        except Exception as e:
            logger.warning("gripper debug write failed: {}", e)

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        return {fkey: {"pos": Array(shape=(7,), dtype=np.float32)} for fkey in self.follower_keys}

    def close(self) -> None:
        logger.info("YamYamBilateralAgent closed")
