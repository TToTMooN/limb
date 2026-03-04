"""
Reusable camera-feed, URDF visualization, and recording monitor backed by a Viser GUI.

Can be given an existing ViserServer (e.g. from YamViserAgent) or will
create its own (standalone mode for GELLO / VR agents).
"""

import os
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import viser
import viser.extras
from loguru import logger

from limb.core.observation import Observation
from limb.sensors.cameras.camera_utils import obs_get_rgb, resize_with_pad


class ViserMonitor:
    """Viser-based monitoring UI for camera feeds, URDF visualization, and recording.

    Parameters
    ----------
    viser_server : optional
        Existing ViserServer to attach GUI elements to.  If *None*, a new
        server is created automatically.
    image_size : int
        Width/height of the square thumbnails shown in the GUI.
    recording_fps : int
        Frame rate written into recorded video files.
    enable_urdf : bool
        If True, load the YAM URDF and show the real robot state in 3-D.
    bimanual : bool
        Whether to visualize two arms (only used when *enable_urdf* is True).
    right_arm_extrinsic : dict | None
        ``{"position": [x,y,z], "rotation": [w,x,y,z]}`` offset for the
        right arm base frame (only used when *bimanual* is True).
    """

    def __init__(
        self,
        viser_server: Optional[viser.ViserServer] = None,
        image_size: int = 224,
        recording_fps: int = 30,
        enable_urdf: bool = False,
        bimanual: bool = False,
        right_arm_extrinsic: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.viser_server = viser_server if viser_server is not None else viser.ViserServer()
        self._image_size = image_size
        self._recording_fps = recording_fps
        self._bimanual = bimanual
        self._right_arm_extrinsic = right_arm_extrinsic

        self._recording = False
        self._writers: Dict[str, cv2.VideoWriter] = {}
        self._record_dir: Optional[Path] = None
        self._lock = threading.Lock()

        self._cam_img_handles: Dict[str, Any] = {}
        self._latest_rgb: Dict[str, np.ndarray] = {}

        # URDF visualization handles (populated by _setup_urdf)
        self._urdf_vis_left: Optional[viser.extras.ViserUrdf] = None
        self._urdf_vis_right: Optional[viser.extras.ViserUrdf] = None

        if enable_urdf:
            self._setup_urdf()

        self._record_button = self.viser_server.gui.add_button("Start Recording", color="green")

        @self._record_button.on_click
        def _(_event: viser.GuiEvent) -> None:
            self._toggle_recording()

    # ------------------------------------------------------------------ #
    #  URDF setup
    # ------------------------------------------------------------------ #

    def _setup_urdf(self) -> None:
        """Load the YAM URDF and add it to the Viser scene."""
        import yourdfpy

        current_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.normpath(
            os.path.join(current_path, "..", "..", "..", "dependencies", "i2rt", "i2rt", "robot_models", "yam", "yam.urdf")
        )
        mesh_dir = os.path.normpath(
            os.path.join(current_path, "..", "..", "..", "dependencies", "i2rt", "i2rt", "robot_models", "yam", "assets")
        )

        urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)

        self.viser_server.scene.add_frame("/base", show_axes=False)
        self._urdf_vis_left = viser.extras.ViserUrdf(self.viser_server, urdf, root_node_name="/base")
        self.viser_server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

        if self._bimanual and self._right_arm_extrinsic is not None:
            right_frame = self.viser_server.scene.add_frame("/base/base_right", show_axes=False)
            right_frame.position = tuple(self._right_arm_extrinsic.get("position", [0, -0.61, 0]))
            if "rotation" in self._right_arm_extrinsic:
                right_frame.wxyz = tuple(self._right_arm_extrinsic["rotation"])
            self._urdf_vis_right = viser.extras.ViserUrdf(
                self.viser_server, deepcopy(urdf), root_node_name="/base/base_right"
            )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def update(self, obs: Any) -> None:
        """Feed a new observation into the monitor (thread-safe).

        Accepts either a typed ``Observation`` (from the main process) or a
        plain dict (when embedded inside an agent subprocess via portal RPC).

        Extracts RGB images, updates the Viser GUI thumbnails, updates URDF
        joint visualization, and writes video frames when recording.
        """
        # --- URDF visualization ---
        if self._urdf_vis_left is not None:
            if isinstance(obs, Observation):
                left = obs.arms.get("left")
                left_jp = left.joint_pos if left is not None else None
                right_data = obs.arms.get("right")
                right_jp = right_data.joint_pos if right_data is not None else None
            else:
                left = obs.get("left")
                left_jp = left["joint_pos"] if isinstance(left, dict) and "joint_pos" in left else None
                right = obs.get("right")
                right_jp = right["joint_pos"] if isinstance(right, dict) and "joint_pos" in right else None
            if left_jp is not None:
                self._urdf_vis_left.update_cfg(np.flip(left_jp[:6]))
            if self._bimanual and self._urdf_vis_right is not None and right_jp is not None:
                self._urdf_vis_right.update_cfg(np.flip(right_jp[:6]))

        # --- Camera feeds ---
        rgb_images = obs_get_rgb(obs)
        if not rgb_images:
            return

        self._latest_rgb = rgb_images

        for cam_name, img in rgb_images.items():
            thumb = resize_with_pad(img, self._image_size, self._image_size)
            if cam_name not in self._cam_img_handles:
                self._cam_img_handles[cam_name] = self.viser_server.gui.add_image(thumb, label=cam_name)
            else:
                self._cam_img_handles[cam_name].image = thumb

        # --- Recording ---
        with self._lock:
            if self._recording:
                for cam_name, img in rgb_images.items():
                    writer = self._writers.get(cam_name)
                    if writer is None:
                        h, w = img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        path = str(self._record_dir / f"{cam_name}.mp4")
                        writer = cv2.VideoWriter(path, fourcc, self._recording_fps, (w, h))
                        self._writers[cam_name] = writer
                    writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        """Release all video writers and stop recording."""
        with self._lock:
            if self._recording:
                for w in self._writers.values():
                    w.release()
                logger.info(f"Recording saved to {self._record_dir}")
                self._writers.clear()
                self._recording = False

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _toggle_recording(self) -> None:
        with self._lock:
            if not self._recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._record_dir = Path("recordings") / f"recording_{ts}"
                self._record_dir.mkdir(parents=True, exist_ok=True)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                for cam_name, img in self._latest_rgb.items():
                    h, w = img.shape[:2]
                    path = str(self._record_dir / f"{cam_name}.mp4")
                    self._writers[cam_name] = cv2.VideoWriter(path, fourcc, self._recording_fps, (w, h))

                self._recording = True
                self._record_button.name = "Stop Recording"
                self._record_button.color = "red"
                logger.info(f"Recording started -> {self._record_dir}")
            else:
                for w in self._writers.values():
                    w.release()
                logger.info(f"Recording saved to {self._record_dir}")
                self._writers.clear()
                self._record_dir = None
                self._recording = False
                self._record_button.name = "Start Recording"
                self._record_button.color = "green"
