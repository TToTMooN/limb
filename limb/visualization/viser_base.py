"""
Abstract base class for bimanual robot IK solving and optional visualization.

Supports headless mode (viser_server=None) for running IK without visualization,
and visualized mode for interactive control via Viser gizmos.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
import yourdfpy

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description as load_urdf_robot_description
except ImportError as e:
    print(f"Error importing robot_descriptions: {e}")
    print("Please run: pip install git+https://github.com/uynitsuj/robot_descriptions.py.git")
    exit()

_AUTO_VISER = object()


@dataclass
class TransformHandle:
    """Data class to store transform handles."""

    tcp_offset_frame: viser.FrameHandle
    control: Optional[viser.TransformControlsHandle] = None


def set_min_distance_from_limits(urdf: yourdfpy.URDF, min_distance_from_limits: float = 0.15) -> yourdfpy.URDF:
    """
    Set the minimum distance from limits for the robot.
    min_distance_from_limits: float in radians
    """
    for joint in urdf.robot.joints:
        if joint.type == "revolute" and joint.limit is not None:
            if joint.limit.lower is not None and joint.limit.upper is not None:
                joint.limit.lower = joint.limit.lower + min_distance_from_limits
                joint.limit.upper = joint.limit.upper - min_distance_from_limits
    return urdf


class ViserAbstractBase(ABC):
    """
    Abstract base class for bimanual robot IK solving.

    Supports two modes:
      - **Visualized** (default): creates or uses a ViserServer with gizmos and URDF overlay.
      - **Headless** (viser_server=None): pure IK, no visualization.  Use set_target() /
        get_target() to drive IK targets programmatically.

    Subclasses must implement solve_ik, _setup_solver_specific,
    _initialize_transform_handles, and home.
    """

    def __init__(
        self,
        rate: float = 100.0,
        viser_server=_AUTO_VISER,
        robot_description: str = "yam_description",
        bimanual: bool = False,
        coordinate_frame: Literal["base", "world"] = "base",
    ):
        self.rate = rate
        self.bimanual = bimanual
        self.coordinate_frame = coordinate_frame

        if robot_description == "yam_description":
            from limb import ROOT_PATH

            yam_models = os.path.join(ROOT_PATH, "dependencies", "i2rt", "i2rt", "robot_models", "yam")
            urdf_path = os.path.join(yam_models, "yam.urdf")
            mesh_dir = os.path.join(yam_models, "assets")
            self.urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
        else:
            self.urdf = set_min_distance_from_limits(
                load_urdf_robot_description(robot_description), min_distance_from_limits=0.25
            )

        # _AUTO_VISER -> create server; explicit None -> headless
        if viser_server is _AUTO_VISER:
            self.viser_server: Optional[viser.ViserServer] = viser.ViserServer()
        else:
            self.viser_server = viser_server

        self.joints: Dict[str, np.ndarray] = {"left": np.zeros(6)}
        if bimanual:
            self.joints["right"] = np.zeros(6)

        # EE targets as plain data -- single source of truth for IK.
        # In viser mode, gizmo handles are synced to / from these values.
        default_wxyz = np.array(vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, np.pi / 2).wxyz, dtype=np.float64)
        sides = ["left", "right"] if bimanual else ["left"]
        self._ee_targets: Dict[str, Dict[str, np.ndarray]] = {}
        for s in sides:
            self._ee_targets[s] = {
                "position": np.array([0.25, 0.0, 0.26], dtype=np.float64),
                "wxyz": default_wxyz.copy(),
                "tcp_offset_pos": np.array([0.0, 0.04, -0.13], dtype=np.float64),
                "tcp_offset_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            }

        self._setup_solver_specific()

        if self.viser_server is not None:
            self._setup_visualization()
            self._setup_gui()
            self._setup_transform_handles()

    # ------------------------------------------------------------------ #
    #  Viser setup (only when viser_server is not None)
    # ------------------------------------------------------------------ #

    def _setup_visualization(self):
        """Setup basic visualization elements."""
        assert self.viser_server is not None
        self.base_frame = self.viser_server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis_left = viser.extras.ViserUrdf(self.viser_server, self.urdf, root_node_name="/base")
        self.ground_grid = self.viser_server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    def _setup_gui(self):
        """Setup enhanced GUI elements."""
        assert self.viser_server is not None
        self.timing_handle = self.viser_server.gui.add_number("Time (ms)", 0.01, disabled=True)
        self.tf_size_handle = self.viser_server.gui.add_slider(
            "Gizmo size", min=0.05, max=0.4, step=0.01, initial_value=0.2
        )
        self.coordinate_frame_handle = self.viser_server.gui.add_dropdown(
            "Coordinate Frame", options=["base", "world"], initial_value=self.coordinate_frame
        )
        self.reset_button = self.viser_server.gui.add_button("Reset to Rest Pose")

    def _setup_transform_handles(self):
        """Setup transform handles for end effectors."""
        assert self.viser_server is not None
        self.transform_handles: Dict[str, TransformHandle] = {
            "left": TransformHandle(
                tcp_offset_frame=self.viser_server.scene.add_frame(
                    "target_left/tcp_offset", show_axes=False, position=(0.0, 0.0, 0.0), wxyz=(1, 0, 0, 0)
                ),
                control=self.viser_server.scene.add_transform_controls("target_left", scale=self.tf_size_handle.value),
            ),
        }
        if self.bimanual:
            self.transform_handles["right"] = TransformHandle(
                tcp_offset_frame=self.viser_server.scene.add_frame(
                    "target_right/tcp_offset", show_axes=False, position=(0.0, 0.0, 0.0), wxyz=(1, 0, 0, 0)
                ),
                control=self.viser_server.scene.add_transform_controls(
                    "target_right", scale=self.tf_size_handle.value
                ),
            )

        @self.tf_size_handle.on_update
        def update_tf_size(_):
            for handle in self.transform_handles.values():
                if handle.control:
                    handle.control.scale = self.tf_size_handle.value
            self._update_optional_handle_sizes()

        self._initialize_transform_handles()
        self._sync_gizmos_to_targets()

    # ------------------------------------------------------------------ #
    #  EE target API (works in both headless and viser modes)
    # ------------------------------------------------------------------ #

    def set_target(self, side: str, position, wxyz) -> None:
        """Set the IK target for one arm.  Syncs to viser gizmo if active."""
        self._ee_targets[side]["position"] = np.asarray(position, dtype=np.float64)
        self._ee_targets[side]["wxyz"] = np.asarray(wxyz, dtype=np.float64)
        if self.viser_server is not None and hasattr(self, "transform_handles"):
            handle = self.transform_handles.get(side)
            if handle and handle.control is not None:
                handle.control.position = tuple(float(v) for v in position)
                handle.control.wxyz = tuple(float(v) for v in wxyz)

    def get_target(self, side: str) -> Dict[str, np.ndarray]:
        """Get the current IK target for one arm (position, wxyz, tcp offsets)."""
        return self._ee_targets[side]

    def set_tcp_offset(self, side: str, position, wxyz) -> None:
        """Set the TCP offset for one arm.  Syncs to viser frame handle if active."""
        self._ee_targets[side]["tcp_offset_pos"] = np.asarray(position, dtype=np.float64)
        self._ee_targets[side]["tcp_offset_wxyz"] = np.asarray(wxyz, dtype=np.float64)
        if self.viser_server is not None and hasattr(self, "transform_handles"):
            handle = self.transform_handles.get(side)
            if handle and handle.tcp_offset_frame is not None:
                handle.tcp_offset_frame.position = tuple(float(v) for v in position)
                handle.tcp_offset_frame.wxyz = tuple(float(v) for v in wxyz)

    def _sync_gizmos_to_targets(self) -> None:
        """Read viser gizmo positions back into _ee_targets (for mouse-drag input)."""
        if not hasattr(self, "transform_handles"):
            return
        for side, handle in self.transform_handles.items():
            if handle.control is not None:
                self._ee_targets[side]["position"] = np.array(handle.control.position, dtype=np.float64)
                self._ee_targets[side]["wxyz"] = np.array(handle.control.wxyz, dtype=np.float64)
            if handle.tcp_offset_frame is not None:
                self._ee_targets[side]["tcp_offset_pos"] = np.array(handle.tcp_offset_frame.position, dtype=np.float64)
                self._ee_targets[side]["tcp_offset_wxyz"] = np.array(handle.tcp_offset_frame.wxyz, dtype=np.float64)

    @property
    def urdf_joint_names(self):
        """Get URDF joint names."""
        return self.urdf.joint_names

    @abstractmethod
    def _update_optional_handle_sizes(self):
        """Override in subclasses to update optional handle sizes."""
        pass

    # ------------------------------------------------------------------ #
    #  IK target resolution and visualization
    # ------------------------------------------------------------------ #

    def update_visualization(self):
        """Update visualization with current state."""
        if self.viser_server is None:
            return
        self.urdf_vis_left.update_cfg(self.joints["left"])

    def get_target_poses(self):
        """Get target poses with TCP offset applied.

        In viser mode, syncs gizmo positions to _ee_targets first so that
        mouse-drag updates are picked up by the IK solver.
        """
        if self.viser_server is not None:
            self._sync_gizmos_to_targets()

        target_poses = {}
        for side, t in self._ee_targets.items():
            control_tf = vtf.SE3(np.array([*t["wxyz"], *t["position"]]))
            tcp_tf = vtf.SE3(np.array([*t["tcp_offset_wxyz"], *t["tcp_offset_pos"]]))
            target_poses[side] = control_tf @ tcp_tf
        return target_poses

    def set_ee_targets(self, left_wxyz_xyz: np.ndarray, right_wxyz_xyz: np.ndarray):
        """
        Set end effector targets.
        left_wxyz_xyz: [wxyz, xyz]
        right_wxyz_xyz: [wxyz, xyz]
        """
        self.set_target("left", left_wxyz_xyz[4:], left_wxyz_xyz[:4])
        if self.bimanual:
            self.set_target("right", right_wxyz_xyz[4:], right_wxyz_xyz[:4])

    def set_tcp_offsets(self, left_offset: np.ndarray, right_offset: Optional[np.ndarray] = None):
        """
        Set TCP offset frames.
        left_offset: [wxyz, xyz]
        right_offset: [wxyz, xyz]
        """
        self.set_tcp_offset("left", left_offset[4:], left_offset[:4])
        if self.bimanual and right_offset is not None:
            self.set_tcp_offset("right", right_offset[4:], right_offset[:4])

    @abstractmethod
    def home(self):
        """Reset robot to rest pose.  Must be implemented by subclasses."""
        raise NotImplementedError

    def run(self):
        """Main run loop with enhanced timing."""
        while True:
            start_time = time.time()
            self.solve_ik()

            if self.viser_server is not None:
                self.update_visualization()

            elapsed_time = time.time() - start_time
            if self.viser_server is not None and hasattr(self, "timing_handle"):
                self.timing_handle.value = 0.99 * self.timing_handle.value + 0.01 * (elapsed_time * 1000)

            sleep_time = max(0, (1.0 / self.rate) - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def _setup_solver_specific(self):
        """Setup solver-specific components.  Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _initialize_transform_handles(self):
        """Initialize transform handle positions.  Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def solve_ik(self):
        """Solve inverse kinematics.  Must be implemented by subclasses."""
        raise NotImplementedError
