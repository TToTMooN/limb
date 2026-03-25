"""Camera feed panel — displays camera thumbnails in the viser sidebar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import viser

from limb.sensors.cameras.camera_utils import obs_get_rgb, resize_with_pad


@dataclass
class CameraPanel:
    """Shows camera RGB thumbnails in the viser GUI sidebar.

    Implements ObservablePanel: attach(server), update(obs), detach().
    """

    image_size: int = 224

    _server: Optional[viser.ViserServer] = field(default=None, init=False, repr=False)
    _cam_img_handles: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _latest_rgb: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def attach(self, server: viser.ViserServer) -> None:
        self._server = server

    def update(self, obs: Any) -> None:
        """Extract RGB images from observation and update thumbnails."""
        if self._server is None:
            return
        rgb_images = obs_get_rgb(obs)
        if not rgb_images:
            return

        self._latest_rgb = rgb_images

        for cam_name, img in rgb_images.items():
            thumb = resize_with_pad(img, self.image_size, self.image_size)
            if cam_name not in self._cam_img_handles:
                self._cam_img_handles[cam_name] = self._server.gui.add_image(thumb, label=cam_name)
            else:
                self._cam_img_handles[cam_name].image = thumb

    def get_latest_rgb(self) -> Dict[str, np.ndarray]:
        """Get the most recent RGB images (used by RecordingPanel)."""
        return self._latest_rgb

    def detach(self) -> None:
        self._cam_img_handles.clear()
        self._latest_rgb.clear()
