"""
Bimanual YAM arms Inverse Kinematics using Pink (Pinocchio-based QP IK) with ViserAbstractBase.

Pink solves differential IK via weighted tasks and QP optimization, producing
joint velocities that are integrated each step. This gives smoother trajectories
compared to per-frame global IK solvers.

Requires:
  - pin-pink (``pip install pin-pink``)
  - A QP solver backend (e.g. ``pip install qpsolvers[quadprog]`` or daqp)
"""

import os
import time
from copy import deepcopy
from typing import Dict, Literal, Optional

import numpy as np
import pink
import pinocchio as pin
import viser
import viser.extras
import viser.transforms as vtf
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

from limb.visualization.viser_base import _AUTO_VISER, TransformHandle, ViserAbstractBase

TARGET_LINK = "link_6"


def _best_qp_solver() -> str:
    import qpsolvers

    if "daqp" in qpsolvers.available_solvers:
        return "daqp"
    return qpsolvers.available_solvers[0]


def _load_pinocchio_model(urdf_path: str, package_dir: str) -> pin.RobotWrapper:
    return pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=[package_dir],
        root_joint=None,
    )


class YamPink(ViserAbstractBase):
    """
    YAM robot IK using Pink (Pinocchio + QP).

    Each arm gets its own pinocchio model, pink Configuration, FrameTask,
    and PostureTask.  The solve_ik loop (called from ViserAbstractBase.run at
    ``rate`` Hz) reads Viser transform-handle targets, solves for a joint
    velocity, integrates it, and writes the result to ``self.joints``.
    """

    def __init__(
        self,
        rate: float = 100.0,
        viser_server=_AUTO_VISER,
        bimanual: bool = False,
        coordinate_frame: Literal["base", "world"] = "base",
        position_cost: float = 50.0,
        orientation_cost: float = 10.0,
        posture_cost: float = 1e-3,
        lm_damping: float = 1.0,
    ):
        self.target_link_names = [TARGET_LINK]
        self.joints: Dict[str, np.ndarray] = {"left": np.zeros(6)}
        self.coordinate_frame = coordinate_frame
        self._dt = 1.0 / rate
        self._position_cost = position_cost
        self._orientation_cost = orientation_cost
        self._posture_cost = posture_cost
        self._lm_damping = lm_damping

        self.robot_pin: Dict[str, pin.RobotWrapper] = {}
        self.configurations: Dict[str, pink.Configuration] = {}
        self.ee_tasks: Dict[str, FrameTask] = {}
        self.posture_tasks: Dict[str, PostureTask] = {}

        if bimanual:
            self.target_link_names = self.target_link_names * 2
            self.joints["right"] = np.zeros(6)

        super().__init__(rate, viser_server, bimanual=bimanual)

    # ------------------------------------------------------------------ #
    #  ViserAbstractBase hooks
    # ------------------------------------------------------------------ #

    def _setup_solver_specific(self):
        """Load pinocchio model and create pink tasks for each arm."""
        current_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.normpath(
            os.path.join(current_path, "..", "..", "..", "dependencies", "i2rt", "i2rt", "robot_models", "yam", "yam.urdf")
        )
        package_dir = os.path.normpath(
            os.path.join(current_path, "..", "..", "..", "dependencies", "i2rt", "i2rt", "robot_models", "yam")
        )

        self._solver = _best_qp_solver()

        sides = ["left", "right"] if self.bimanual else ["left"]
        for side in sides:
            robot = _load_pinocchio_model(urdf_path, package_dir)
            config = pink.Configuration(robot.model, robot.data, robot.q0)

            ee_task = FrameTask(
                TARGET_LINK,
                position_cost=self._position_cost,
                orientation_cost=self._orientation_cost,
                lm_damping=self._lm_damping,
            )
            posture_task = PostureTask(cost=self._posture_cost)

            ee_task.set_target_from_configuration(config)
            posture_task.set_target_from_configuration(config)

            self.robot_pin[side] = robot
            self.configurations[side] = config
            self.ee_tasks[side] = ee_task
            self.posture_tasks[side] = posture_task

        self.rest_pose = np.flip(self.configurations["left"].q).copy()

    def _setup_visualization(self):
        super()._setup_visualization()
        assert self.viser_server is not None
        if self.bimanual:
            self.base_frame_right = self.viser_server.scene.add_frame("/base/base_right", show_axes=False)
            self.base_frame_right.position = (0.0, -0.61, 0.0)
            self.urdf_vis_right = viser.extras.ViserUrdf(
                self.viser_server, deepcopy(self.urdf), root_node_name="/base/base_right"
            )

    def _setup_gui(self):
        super()._setup_gui()
        assert self.viser_server is not None
        self.timing_handle_left = self.viser_server.gui.add_number("Left Arm Time (ms)", 0.01, disabled=True)
        if self.bimanual:
            self.timing_handle_right = self.viser_server.gui.add_number("Right Arm Time (ms)", 0.01, disabled=True)

    def _initialize_transform_handles(self):
        """Set initial IK target poses for the arms (same as YamPyroki)."""
        assert self.viser_server is not None
        if self.transform_handles["left"].control is not None:
            self.transform_handles["left"].control.position = (0.25, 0.0, 0.26)
            self.transform_handles["left"].control.wxyz = vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, np.pi / 2).wxyz
            self.transform_handles["left"].tcp_offset_frame.position = (0.0, 0.04, -0.13)

        if self.bimanual:
            if self.transform_handles["right"].control is not None:
                self.transform_handles["right"].control.remove()
                self.transform_handles["right"].tcp_offset_frame.remove()
            self.transform_handles["right"] = TransformHandle(
                tcp_offset_frame=self.viser_server.scene.add_frame(
                    "/base/base_righttarget_right/tcp_offset",
                    show_axes=False,
                    position=(0.0, 0.04, -0.13),
                    wxyz=vtf.SO3.from_rpy_radians(0.0, 0.0, 0.0).wxyz,
                ),
                control=self.viser_server.scene.add_transform_controls(
                    "/base/base_right/target_right",
                    scale=self.tf_size_handle.value,
                    position=(0.25, 0.0, 0.26),
                    wxyz=vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, np.pi / 2).wxyz,
                ),
            )

    def _update_optional_handle_sizes(self):
        pass

    # ------------------------------------------------------------------ #
    #  IK solving
    # ------------------------------------------------------------------ #

    def solve_ik(self):
        """Solve differential IK for each arm using pink."""
        target_poses = self.get_target_poses()

        if self.bimanual:
            if "left" not in target_poses or "right" not in target_poses:
                return
        elif "left" not in target_poses:
            return

        for side in target_poses:
            target_tf = target_poses[side]
            pos = np.asarray(target_tf.translation(), dtype=np.float64)
            wxyz = target_tf.rotation().wxyz
            R = vtf.SO3(wxyz).as_matrix()
            target_se3 = pin.SE3(np.asarray(R, dtype=np.float64), pos)

            self.ee_tasks[side].set_target(target_se3)

            config = self.configurations[side]
            tasks = [self.ee_tasks[side], self.posture_tasks[side]]

            start = time.time()
            velocity = solve_ik(config, tasks, self._dt, solver=self._solver)
            config.integrate_inplace(velocity, self._dt)
            elapsed_ms = (time.time() - start) * 1000

            self.joints[side] = np.flip(np.array(config.q, dtype=np.float64)).copy()

            if side == "left" and hasattr(self, "timing_handle_left"):
                self.timing_handle_left.value = 0.9 * self.timing_handle_left.value + 0.1 * elapsed_ms
            elif side == "right" and hasattr(self, "timing_handle_right"):
                self.timing_handle_right.value = 0.9 * self.timing_handle_right.value + 0.1 * elapsed_ms

    # ------------------------------------------------------------------ #
    #  Visualization
    # ------------------------------------------------------------------ #

    def update_visualization(self):
        if self.viser_server is None or self.joints is None:
            return
        self.urdf_vis_left.update_cfg(self.joints["left"])
        if self.bimanual:
            self.urdf_vis_right.update_cfg(self.joints["right"])

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def home(self):
        """Reset all arms to the URDF rest pose."""
        pin_rest = np.flip(self.rest_pose).copy()
        sides = ["left", "right"] if self.bimanual else ["left"]
        for side in sides:
            self.joints[side] = self.rest_pose.copy()
            self.configurations[side].update(pin_rest)
            self.ee_tasks[side].set_target_from_configuration(self.configurations[side])

        default_wxyz = np.array(vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, np.pi / 2).wxyz, dtype=np.float64)
        for side in sides:
            self._ee_targets[side]["position"] = np.array([0.25, 0.0, 0.26], dtype=np.float64)
            self._ee_targets[side]["wxyz"] = default_wxyz.copy()

        if self.viser_server is not None:
            self._initialize_transform_handles()
            self.urdf_vis_left.update_cfg(self.rest_pose)
            if self.bimanual:
                self.urdf_vis_right.update_cfg(self.rest_pose)

    def get_joint_positions(self) -> Optional[np.ndarray]:
        if self.bimanual:
            if self.joints["left"] is not None and self.joints["right"] is not None:
                return np.concatenate([self.joints["left"], self.joints["right"]])
            return None
        elif self.joints["left"] is not None:
            return self.joints["left"]
        return None


def main():
    """Standalone test: launch bimanual YAM Pink IK with Viser."""
    viz = YamPink(rate=100.0, bimanual=True)
    viz.run()


if __name__ == "__main__":
    main()
