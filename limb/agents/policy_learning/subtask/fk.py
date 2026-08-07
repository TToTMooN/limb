"""EE-pose FK injection for the sub-task loop (review C1).

The YAM driver returns {joint_pos, joint_vel, joint_eff, gripper_pos} only; ``ee_pose``
is filled by IK/teleop agents, NOT by the policy-learning launch path — so the vials
verifier/reset (which read ``obs['right']['ee_pose']``) would wedge the loop. This
injector computes forward kinematics with pinocchio (already a limb dep via pin-pink)
and fills ``ee_pose`` in-place when missing.

Frame convention: ee_pose = [qw, qx, qy, qz, x, y, z] (limb/core/observation.py), in the
ARM'S OWN BASE frame — the same frame table_z / PICKUP_REGION are calibrated in, so the
verifier/reset geometry stays consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_URDF = "dependencies/i2rt/i2rt/robot_models/yam/yam.urdf"
DEFAULT_EE_FRAME = "link_6"          # same target link yam_pink IK uses


@dataclass
class EEPoseInjector:
    """Fills obs[side]['ee_pose'] via pinocchio FK when the driver didn't provide it."""

    urdf_path: str = DEFAULT_URDF
    ee_frame: str = DEFAULT_EE_FRAME
    sides: List[str] = field(default_factory=lambda: ["right"])
    overwrite: bool = False           # True: recompute even if ee_pose already present

    def __post_init__(self) -> None:
        self._model = None
        self._data = None
        self._frame_id = None
        self._warned = False

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            import pathlib

            import pinocchio as pin
            path = pathlib.Path(self.urdf_path)
            if not path.is_absolute():
                # resolve relative to the limb repo root (…/limb/agents/policy_learning/subtask/fk.py)
                path = pathlib.Path(__file__).resolve().parents[4] / self.urdf_path
            self._model = pin.buildModelFromUrdf(str(path))
            self._data = self._model.createData()
            self._frame_id = self._model.getFrameId(self.ee_frame)
            if self._frame_id >= self._model.nframes:
                raise ValueError(f"frame {self.ee_frame!r} not in {path.name}")
            return True
        except Exception as e:
            if not self._warned:
                self._warned = True
                try:
                    from loguru import logger
                    logger.warning("[EEPoseInjector] disabled — could not build FK model: {}", e)
                except Exception:
                    pass
            self._model = None
            return False

    def ee_pose(self, joint_pos: np.ndarray) -> Optional[np.ndarray]:
        """FK for one arm: joint_pos (>=6,) -> [qw,qx,qy,qz, x,y,z] in the arm base frame."""
        if not self._ensure_model():
            return None
        import pinocchio as pin
        q = np.zeros(self._model.nq)
        j = np.asarray(joint_pos, dtype=float).reshape(-1)
        n = min(len(j), self._model.nq)
        q[:n] = j[:n]
        pin.framesForwardKinematics(self._model, self._data, q)
        M = self._data.oMf[self._frame_id]
        quat = pin.Quaternion(M.rotation)                 # (x, y, z, w) internally
        return np.array([quat.w, quat.x, quat.y, quat.z,
                         M.translation[0], M.translation[1], M.translation[2]], dtype=np.float32)

    def position_jacobian(self, joint_pos: np.ndarray) -> Optional[np.ndarray]:
        """(3, 6) world-aligned POSITION rows of the EE frame jacobian at joint_pos —
        for small servo tasks (e.g. the HUMAN-phase view lift, user 2026-07-27)."""
        if not self._ensure_model():
            return None
        import pinocchio as pin
        q = np.zeros(self._model.nq)
        j = np.asarray(joint_pos, dtype=float).reshape(-1)
        n = min(len(j), self._model.nq)
        q[:n] = j[:n]
        pin.computeJointJacobians(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        J = pin.getFrameJacobian(self._model, self._data, self._frame_id,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return np.asarray(J[:3, :6], dtype=np.float64)

    def inject(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Fill obs[side]['ee_pose'] in place (no copy) for each configured side."""
        for side in self.sides:
            try:
                arm = obs[side]
            except Exception:
                continue
            has = arm.get("ee_pose") is not None
            if has and not self.overwrite:
                continue
            try:
                pose = self.ee_pose(arm["joint_pos"])
            except Exception:
                pose = None
            if pose is not None:
                arm["ee_pose"] = pose
        return obs
