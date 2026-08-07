"""Non-blocking IK-servo control primitives for the coding-agent reset (review H16).

The authored reset (artifacts/vials_grasp/reset_policy.py) is a phase machine that calls
``move_right_ee_to(target)`` then ``return hold()`` every tick. The review's H16 problem:
if those primitives block or command the robot out-of-band, they either stall the 30 Hz
loop or race the env's hold — two writers to the robot.

This module makes the primitives PER-TICK and side-effect-free w.r.t. the robot:
  - ``begin_tick(obs)`` is called once per control step (by the wrapper in
    vials_artifacts) with the CURRENT observation;
  - ``move_right_ee_to(pos)`` computes ONE damped-least-squares IK step toward the
    target and stores it as this tick's PENDING action (it does not move anything);
  - ``open/close_right_gripper()`` set the gripper command on the pending action;
  - ``hold()`` returns the pending action (or ``{}`` = env holds) — so the action flows
    out through the reset policy's normal ``act(obs) -> action`` return value, at the
    loop rate, with per-joint step clipping and joint-limit clamping. ONE writer.

Units: joint targets in rad; gripper command in ACTION units (0 = closed/holding the
vial — the inverse reset starts with the vial held — 2.2 = open). Observations use the
0-1 measured width scale and are never fed back as commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .fk import DEFAULT_EE_FRAME, DEFAULT_URDF

# Canonical TABLE-FACING wrist orientation (link_6 rotation, right-arm base frame),
# CALIBRATED FROM HUMAN PUT-BACK DEMOS (session 231204, 4 FAILURE-labeled episodes,
# 185 vial-held frames; orientation spread p95 = 0.28 rad). In this pose the wrist
# camera looks AT THE TABLE — the pose humans hold while carrying/placing a vial.
# The reset anchors its orientation hold here (user rule 2026-07-07: during reset the
# wrist camera must face the table so the coding agents can see the scene).
_R_TABLE_FACING = np.array([
    [0.188092, 0.481338, 0.856117],
    [0.982134, -0.087034, -0.166846],
    [-0.005798, 0.872204, -0.489109],
])


@dataclass
class ResetControl:
    urdf_path: str = DEFAULT_URDF
    ee_frame: str = DEFAULT_EE_FRAME
    step_rad: float = 0.02            # max per-joint step per tick (0.02 rad @30 Hz = 0.6 rad/s)
    pos_tol: float = 0.008            # within this of the target -> deadband (hold)
    damping: float = 0.05             # DLS lambda
    ori_weight: float = 0.5           # orientation-task weight in the 6-DOF DLS
    # Command INTEGRATION (stiction fix, on-robot 2026-07-06 22:24 run): stepping the
    # target from the OBSERVED joints each tick keeps the PD error at step_rad (0.01
    # rad) forever — below the arms' breakaway friction under load, so the robot never
    # moved in ANY reset run (EE frozen +/-0.1 mm with commands flowing). The target
    # must instead advance from the last COMMANDED position so the PD error can build
    # until the joint breaks away — capped at lead_max ahead of the observed joints
    # (VLA chunks routinely command similar leads).
    lead_max: float = 0.08
    gripper_open_cmd: float = 2.2     # ACTION units
    gripper_close_cmd: float = 0.0
    z_floor: Optional[float] = None   # never command the EE below this z (set to table_z - 0.01)

    def __post_init__(self) -> None:
        self._model = None
        self._data = None
        self._fid = None
        self._obs: Optional[Dict[str, Any]] = None
        self._pending: Optional[Dict[str, Any]] = None
        self._q_cmd_prev: Optional[np.ndarray] = None   # last commanded q6 (integration state)
        self._grip_cmd = self.gripper_close_cmd     # reset precondition: vial HELD -> stay closed
        self._warned = False
        # EE orientation HELD by the 6-DOF IK during reset motions: ALWAYS the demo-
        # calibrated TABLE-FACING pose. The reset is ADAPTIVE to the entry state
        # (user rule 2026-07-07): whatever twist the RL episode left on the wrist is
        # actively rotated out during the reset, so every placement releases upright
        # and every RL episode starts from the same canonical wrist pose with the
        # wrist camera on the table. The earlier keep-entry-pose-if-close logic froze
        # residual twists in place (recording 005054 ep1: the wrist ran a whole
        # episode 0.99 rad off table-facing — the vial was barely graspable).
        self._R_star = None

    # -- lifecycle -------------------------------------------------------------
    def begin_tick(self, obs: Dict[str, Any]) -> None:
        self._obs = obs
        self._pending = None
        if self._R_star is None and self._ensure_model():
            self._R_star = _R_TABLE_FACING.copy()

    def reset(self) -> None:
        self._grip_cmd = self.gripper_close_cmd     # next reset starts with the vial held again
        self._pending = None
        self._R_star = None                          # recapture the carry orientation on entry
        self._q_cmd_prev = None                      # restart command integration from observed

    def primitives(self) -> Dict[str, Any]:
        """The callables injected into the authored reset's namespace."""
        return {
            "move_right_ee_to": self.move_right_ee_to,
            "open_right_gripper": self.open_right_gripper,
            "close_right_gripper": self.close_right_gripper,
            "hold": self.hold,
            "ori_error": self.ori_error,
            "tilt_error": self.tilt_error,
        }

    def tilt_error(self) -> float:
        """Angle (rad) between the CURRENT and canonical TOOL Z-AXIS — pure tilt,
        ignoring yaw about the vertical. The release gate uses THIS: a yawed wrist
        releases a vial upright (harmless), a pitched/rolled one tips it over
        (on-robot 23:34 drop: full-rotation gate passed at 0.28-0.39 rad that was
        mostly tilt)."""
        R = self._current_rotation()
        if R is None or self._R_star is None:
            return 0.0
        c = float(np.clip(np.dot(self._R_star[:, 2], R[:, 2]), -1.0, 1.0))
        return float(np.arccos(c))

    def _current_rotation(self):
        if self._obs is None or not self._ensure_model():
            return None
        try:
            import pinocchio as pin
            q = np.asarray(self._obs["right"]["joint_pos"], dtype=float).reshape(-1)[:6]
            qf = np.zeros(self._model.nq); qf[:len(q)] = q
            pin.framesForwardKinematics(self._model, self._data, qf)
            return np.array(self._data.oMf[self._fid].rotation)
        except Exception:
            return None

    def ori_error(self) -> float:
        """Angle (rad) between the CURRENT EE orientation and the canonical tip-down
        orientation. The reset gates the gripper-OPEN on this: never release with the
        wrist rotated away (that is how the vial got dropped horizontal at height)."""
        if self._obs is None or self._R_star is None or not self._ensure_model():
            return 0.0
        try:
            import pinocchio as pin
            q = np.asarray(self._obs["right"]["joint_pos"], dtype=float).reshape(-1)[:6]
            qf = np.zeros(self._model.nq); qf[:len(q)] = q
            pin.framesForwardKinematics(self._model, self._data, qf)
            return float(np.linalg.norm(pin.log3(self._R_star @ np.array(self._data.oMf[self._fid].rotation).T)))
        except Exception:
            return 0.0

    # -- primitives (called by the AUTHORED code) --------------------------------
    def move_right_ee_to(self, pos, max_steps: int = 60) -> None:
        """Register ONE IK-servo step toward world `pos` (x,y,z in the right-arm base
        frame) as this tick's pending action. `max_steps` kept for signature
        compatibility with the sim API; pacing comes from the authored phase machine
        calling this every tick until its own convergence check passes."""
        if self._obs is None or not self._ensure_model():
            return
        try:
            q = np.asarray(self._obs["right"]["joint_pos"], dtype=float).reshape(-1)[:6]
        except Exception:
            return
        target = np.asarray(pos, dtype=float).reshape(3)
        if self.z_floor is not None:
            target[2] = max(target[2], self.z_floor)

        import pinocchio as pin
        qf = np.zeros(self._model.nq); qf[:len(q)] = q
        pin.framesForwardKinematics(self._model, self._data, qf)
        M = self._data.oMf[self._fid]
        cur = np.asarray(M.translation)
        pos_err = target - cur
        # orientation error toward the canonical tip-down orientation (6-DOF task)
        if self._R_star is not None:
            rot_err = pin.log3(self._R_star @ np.array(M.rotation).T)
        else:
            rot_err = np.zeros(3)
        if np.linalg.norm(pos_err) < self.pos_tol and np.linalg.norm(rot_err) < 0.05:
            self._q_cmd_prev = q.copy()
            self._set_pending(q)                     # deadband: hold this pose
            return
        pin.computeJointJacobians(self._model, self._data, qf)
        J6 = pin.getFrameJacobian(self._model, self._data, self._fid,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, :len(q)]
        w = self.ori_weight
        err6 = np.concatenate([pos_err, w * rot_err])
        Jw = np.vstack([J6[:3], w * J6[3:]])
        lam2 = self.damping ** 2
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam2 * np.eye(6), err6)
        dq = np.clip(dq, -self.step_rad, self.step_rad)
        # Integrate from the last COMMANDED position, not the observed one (stiction
        # fix, see lead_max above) — but never lead the observed joints by more than
        # lead_max, and keep the error/Jacobian evaluated at the OBSERVED pose so the
        # step direction always reflects the real kinematic state.
        q_ref = q if self._q_cmd_prev is None else np.clip(
            self._q_cmd_prev, q - self.lead_max, q + self.lead_max)
        q_new = q_ref + dq
        lo = np.asarray(self._model.lowerPositionLimit)[:len(q)]
        hi = np.asarray(self._model.upperPositionLimit)[:len(q)]
        good = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        q_new[good] = np.clip(q_new[good], lo[good], hi[good])
        self._q_cmd_prev = q_new.copy()
        self._set_pending(q_new)

    def open_right_gripper(self) -> None:
        self._grip_cmd = self.gripper_open_cmd
        self._hold_arm_with_grip()

    def close_right_gripper(self) -> None:
        self._grip_cmd = self.gripper_close_cmd
        self._hold_arm_with_grip()

    def hold(self) -> Dict[str, Any]:
        """The authored code's per-tick return: this tick's pending action, or {} (env
        holds all arms). The LEFT arm is never commanded by the reset."""
        return self._pending if self._pending is not None else {}

    # -- internals ----------------------------------------------------------------
    def _hold_arm_with_grip(self) -> None:
        """Pending action = held arm pose + the new gripper command (gripper-only move).
        Holds the last COMMANDED pose when one exists so the servo target isn't snapped
        back to the (lagging) observed joints mid-release."""
        if self._pending is not None:                # keep an arm move made this tick
            self._pending["right"]["pos"][6] = self._grip_cmd
            return
        if self._q_cmd_prev is not None:
            self._set_pending(self._q_cmd_prev)
            return
        try:
            q = np.asarray(self._obs["right"]["joint_pos"], dtype=float).reshape(-1)[:6]
        except Exception:
            return
        self._set_pending(q)

    def _set_pending(self, q6: np.ndarray) -> None:
        self._pending = {"right": {"pos": np.concatenate(
            [np.asarray(q6, np.float32), [np.float32(self._grip_cmd)]])}}

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            import pathlib

            import pinocchio as pin
            path = pathlib.Path(self.urdf_path)
            if not path.is_absolute():
                path = pathlib.Path(__file__).resolve().parents[4] / self.urdf_path
            self._model = pin.buildModelFromUrdf(str(path))
            self._data = self._model.createData()
            self._fid = self._model.getFrameId(self.ee_frame)
            return True
        except Exception as e:
            if not self._warned:
                self._warned = True
                try:
                    from loguru import logger
                    logger.warning("[ResetControl] disabled — no IK model: {}", e)
                except Exception:
                    pass
            return False


@dataclass
class TickWrappedReset:
    """Adapts the authored reset (which calls the primitives) to the ResetPolicy
    protocol: feeds the current obs to the control each tick and passes the pending
    action out through the normal act() return path."""
    inner: Any
    control: ResetControl

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.control.begin_tick(obs)
        return self.inner.act(obs)

    def done(self, obs: Dict[str, Any]) -> bool:
        return self.inner.done(obs)

    def reset(self) -> None:
        self.control.reset()
        self.inner.reset()

    @property
    def failed(self) -> bool:
        return bool(getattr(self.inner, "failed", False))
