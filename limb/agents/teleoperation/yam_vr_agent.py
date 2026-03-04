"""
VR teleoperation agent for bimanual YAM arms using Pico VR controllers.

Replaces the Viser gizmo input with VR controller poses while driving
the IK solver in headless mode (no Viser server).  Camera feeds and
recording are handled by the shared ViserMonitor in launch.py.

IK backend is selectable via the ``ik_solver`` parameter:
  - "pink" (default)   -- Pinocchio QP differential IK (requires pin-pink)
  - "pyroki"           -- JAX-based global IK, no extra deps

Requires:
  - XRoboToolkit PC Service running on the PC
  - xrobotoolkit_sdk installed (bash scripts/install_xrobotoolkit_sdk.sh)
  - Pico headset connected and streaming

VR controller button mapping:
  - A / X  : Reset corresponding arm to initial EE pose (task space)
"""

import threading
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import viser.transforms as vtf
from dm_env.specs import Array
from loguru import logger

from limb.agents.agent import Agent
from limb.agents.teleoperation.yam_viser_agent import _create_ik_solver
from limb.utils.portal_utils import remote
from limb.utils.xr_client import XrClient

# Default VR-to-robot frame rotation.
R_VR_TO_ROBOT_DEFAULT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
], dtype=np.float64)

GRIP_ACTIVATION_THRESHOLD = 0.9
YAM_GRIPPER_OPEN = 0.0
YAM_GRIPPER_CLOSED = 2.4

DEFAULT_INITIAL_EE_POSITION = [0.25, 0.0, 0.26]
DEFAULT_INITIAL_EE_RPY = [np.pi / 2, 0.0, np.pi / 2]

RESET_DURATION_S = 1.5


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions (wxyz)."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def _smoothstep(t: float) -> float:
    """Hermite smoothstep: zero velocity at t=0 and t=1."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _scale_rotation(delta_rot: vtf.SO3, scale: float) -> vtf.SO3:
    """Amplify (or dampen) a rotation delta via axis-angle scaling."""
    wxyz = delta_rot.wxyz
    w = float(wxyz[0])
    xyz = np.asarray(wxyz[1:], dtype=np.float64)
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    if angle < 1e-8:
        return delta_rot
    axis = xyz / np.sin(angle / 2.0)
    scaled_angle = angle * scale
    w_new = np.cos(scaled_angle / 2.0)
    xyz_new = axis * np.sin(scaled_angle / 2.0)
    return vtf.SO3(np.array([w_new, *xyz_new]))


class YamVrAgent(Agent):
    def __init__(
        self,
        bimanual: bool = False,
        right_arm_extrinsic: Optional[Dict[str, Any]] = None,
        scale_factor: float = 1.5,
        rotation_scale_factor: float = 1.5,
        R_vr_to_robot: Optional[np.ndarray] = None,
        ik_solver: str = "pink",
        ik_params: Optional[Dict[str, Any]] = None,
        initial_ee_position: Optional[List[float]] = None,
        initial_ee_rpy: Optional[List[float]] = None,
    ) -> None:
        self.bimanual = bimanual
        self.right_arm_extrinsic = right_arm_extrinsic
        self.scale_factor = scale_factor
        self.rotation_scale_factor = rotation_scale_factor
        self.R_vr_to_robot = R_vr_to_robot if R_vr_to_robot is not None else R_VR_TO_ROBOT_DEFAULT
        self.R_vr_so3 = vtf.SO3.from_matrix(self.R_vr_to_robot)

        self.initial_ee_position = tuple(initial_ee_position or DEFAULT_INITIAL_EE_POSITION)
        self.initial_ee_rpy = tuple(initial_ee_rpy or DEFAULT_INITIAL_EE_RPY)
        self.initial_ee_wxyz = vtf.SO3.from_rpy_radians(*self.initial_ee_rpy).wxyz

        if bimanual:
            assert right_arm_extrinsic is not None, "right_arm_extrinsic must be provided for bimanual robot"

        # Headless IK — no ViserServer
        self.ik = _create_ik_solver(ik_solver, ik_params=ik_params, viser_server=None, bimanual=bimanual)

        # VR state tracking per arm
        self.sides = ["left", "right"] if bimanual else ["left"]
        self.ref_vr_pos: Dict[str, Optional[np.ndarray]] = {s: None for s in self.sides}
        self.ref_vr_rot: Dict[str, Optional[vtf.SO3]] = {s: None for s in self.sides}
        self.ref_ee_pos: Dict[str, Optional[np.ndarray]] = {s: None for s in self.sides}
        self.ref_ee_rot: Dict[str, Optional[vtf.SO3]] = {s: None for s in self.sides}
        self.gripper_value: Dict[str, float] = {s: YAM_GRIPPER_OPEN for s in self.sides}
        self.active: Dict[str, bool] = {s: False for s in self.sides}

        self._reset_btn_prev: Dict[str, bool] = {"left": False, "right": False}
        self._reset_t0: Dict[str, Optional[float]] = {s: None for s in self.sides}
        self._reset_start_pos: Dict[str, Optional[np.ndarray]] = {s: None for s in self.sides}
        self._reset_start_wxyz: Dict[str, Optional[np.ndarray]] = {s: None for s in self.sides}

        self.xr_client = XrClient()

        self.obs = None

        # Set initial IK targets
        self._set_initial_pose()

        # Start IK solver thread
        self.ik_thread = threading.Thread(target=self.ik.run, daemon=True)
        self.ik_thread.start()

        # Start VR input processing thread
        self.vr_thread = threading.Thread(target=self._vr_processing_loop, daemon=True)
        self.vr_thread.start()

    # ------------------------------------------------------------------ #
    #  Pose helpers
    # ------------------------------------------------------------------ #

    def _set_initial_pose(self) -> None:
        """Set IK targets to the configured initial EE pose (startup only)."""
        for side in self.sides:
            self.ik.set_target(side, self.initial_ee_position, self.initial_ee_wxyz)
            self.ref_vr_pos[side] = None
            self.ref_vr_rot[side] = None
            self.ref_ee_pos[side] = None
            self.ref_ee_rot[side] = None
            self.active[side] = False
        logger.info("Set initial EE pose: pos={} rpy={}", self.initial_ee_position, self.initial_ee_rpy)

    def _begin_smooth_reset(self, side: str) -> None:
        """Start a smooth interpolation of the IK target back to the initial pose."""
        target = self.ik.get_target(side)
        self._reset_start_pos[side] = target["position"].copy()
        self._reset_start_wxyz[side] = target["wxyz"].copy()
        self._reset_t0[side] = time.time()
        self.ref_vr_pos[side] = None
        self.ref_vr_rot[side] = None
        self.ref_ee_pos[side] = None
        self.ref_ee_rot[side] = None
        self.active[side] = False
        logger.info("Smooth reset started for {} arm ({:.1f}s)", side, RESET_DURATION_S)

    def _tick_smooth_reset(self, side: str) -> bool:
        """Advance the smooth-reset interpolation for one arm.

        Returns True if the reset is still in progress.
        """
        t0 = self._reset_t0[side]
        if t0 is None:
            return False

        alpha = (time.time() - t0) / RESET_DURATION_S
        finished = alpha >= 1.0
        s = _smoothstep(min(alpha, 1.0))

        start_pos = self._reset_start_pos[side]
        start_wxyz = self._reset_start_wxyz[side]
        target_pos = np.array(self.initial_ee_position)
        target_wxyz = np.array(self.initial_ee_wxyz)

        if start_pos is not None and start_wxyz is not None:
            pos = start_pos * (1.0 - s) + target_pos * s
            wxyz = _quat_slerp(start_wxyz, target_wxyz, s)
            self.ik.set_target(side, pos, wxyz)

        if finished:
            self._reset_t0[side] = None
            self._reset_start_pos[side] = None
            self._reset_start_wxyz[side] = None
            logger.info("Smooth reset complete for {} arm", side)

        return not finished

    # ------------------------------------------------------------------ #
    #  VR input
    # ------------------------------------------------------------------ #

    def _transform_vr_pose(self, vr_pose: np.ndarray) -> tuple[np.ndarray, vtf.SO3]:
        """Transform a VR controller pose from VR frame to robot base frame."""
        pos_robot = self.R_vr_to_robot @ np.array(vr_pose[:3])
        quat_xyzw = vr_pose[3:7]
        rot_vr = vtf.SO3(np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]))
        rot_robot = self.R_vr_so3 @ rot_vr @ self.R_vr_so3.inverse()
        return pos_robot, rot_robot

    def _handle_buttons(self) -> None:
        """Check VR controller buttons and dispatch actions."""
        button_to_side = {"left": self.xr_client.get_button("X"), "right": self.xr_client.get_button("A")}
        for side in self.sides:
            pressed = button_to_side[side]
            if pressed and not self._reset_btn_prev[side]:
                self._begin_smooth_reset(side)
            self._reset_btn_prev[side] = pressed

    def _vr_processing_loop(self) -> None:
        """Read VR controller input and update IK targets at ~100 Hz."""
        while True:
            self._handle_buttons()

            for side in self.sides:
                if self._tick_smooth_reset(side):
                    continue

                grip_val = self.xr_client.get_grip(side)
                trigger_val = self.xr_client.get_trigger(side)
                self.gripper_value[side] = (1.0 - trigger_val) * YAM_GRIPPER_CLOSED

                was_active = self.active[side]
                is_active = grip_val > GRIP_ACTIVATION_THRESHOLD
                self.active[side] = is_active

                if is_active:
                    vr_pose = self.xr_client.get_pose(f"{side}_controller")
                    pos_robot, rot_robot = self._transform_vr_pose(vr_pose)

                    if not was_active:
                        self.ref_vr_pos[side] = pos_robot.copy()
                        self.ref_vr_rot[side] = rot_robot
                        target = self.ik.get_target(side)
                        self.ref_ee_pos[side] = target["position"].copy()
                        self.ref_ee_rot[side] = vtf.SO3(target["wxyz"].copy())
                        continue

                    ref_vr_pos = self.ref_vr_pos[side]
                    ref_vr_rot = self.ref_vr_rot[side]
                    ref_ee_pos = self.ref_ee_pos[side]
                    ref_ee_rot = self.ref_ee_rot[side]
                    if ref_vr_pos is None or ref_vr_rot is None or ref_ee_pos is None or ref_ee_rot is None:
                        continue

                    delta_pos = (pos_robot - ref_vr_pos) * self.scale_factor
                    delta_rot = rot_robot @ ref_vr_rot.inverse()
                    if self.rotation_scale_factor != 1.0:
                        delta_rot = _scale_rotation(delta_rot, self.rotation_scale_factor)

                    new_pos = ref_ee_pos + delta_pos
                    new_rot = delta_rot @ ref_ee_rot

                    self.ik.set_target(side, new_pos, new_rot.wxyz)

                else:
                    if was_active:
                        self.ref_vr_pos[side] = None
                        self.ref_vr_rot[side] = None
                        self.ref_ee_pos[side] = None
                        self.ref_ee_rot[side] = None

            time.sleep(0.01)

    # ------------------------------------------------------------------ #
    #  Agent interface
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Any:
        self.obs = deepcopy(obs)

        action: Dict[str, Dict[str, np.ndarray]] = {
            "left": {
                "pos": np.concatenate([np.flip(self.ik.joints["left"]), [self.gripper_value["left"]]]),
            }
        }
        if self.bimanual:
            action["right"] = {
                "pos": np.concatenate([np.flip(self.ik.joints["right"]), [self.gripper_value["right"]]]),
            }

        return action

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        """Define the action specification."""
        action_spec: Dict[str, Dict[str, Array]] = {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
        }
        if self.bimanual:
            action_spec["right"] = {"pos": Array(shape=(7,), dtype=np.float32)}
        return action_spec
