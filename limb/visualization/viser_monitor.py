"""ViserMonitor — backward-compatible wrapper around ViserApp + panels.

Preserved for backward compatibility with existing configs and code that
creates ViserMonitor directly. New code should use ViserApp + individual
panels instead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import viser

from limb.visualization.app import ViserApp
from limb.visualization.panels.camera_panel import CameraPanel
from limb.visualization.panels.recording_panel import RecordingPanel
from limb.visualization.panels.urdf_panel import URDFPanel


class ViserMonitor:
    """Backward-compatible wrapper that composes CameraPanel + URDFPanel + RecordingPanel.

    New code should use ViserApp + panels directly. This class exists so that
    external code (e.g. scripts, custom agents) that creates ViserMonitor
    continues to work.
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
        self._app = ViserApp(viser_server=viser_server) if viser_server is not None else ViserApp()
        self.viser_server = self._app.server

        self._camera_panel = CameraPanel(image_size=image_size)
        self._app.add_panel("cameras", self._camera_panel)

        if enable_urdf:
            self._app.add_panel("urdf", URDFPanel(bimanual=bimanual, right_arm_extrinsic=right_arm_extrinsic))

        self._recording_panel = RecordingPanel(
            recording_fps=recording_fps,
            get_latest_rgb=self._camera_panel.get_latest_rgb,
        )
        self._app.add_panel("recording", self._recording_panel)

    def update(self, obs: Any) -> None:
        """Feed a new observation into all panels."""
        self._app.update(obs)
        # Also feed frames to recording panel
        rgb = self._camera_panel.get_latest_rgb()
        if rgb:
            self._recording_panel.update_frame(rgb)

    def close(self) -> None:
        self._app.close()
