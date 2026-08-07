"""RealSense camera driver — re-exported from robocam for backward compatibility.

New code should import directly from ``robocam.drivers.realsense``.
"""

from dataclasses import dataclass

import numpy as np
from robocam.drivers.realsense import RealsenseCamera, discover_devices  # noqa: F401


@dataclass
class RealsenseDepthLite(RealsenseCamera):
    """RealsenseCamera whose aligned depth frame is downscaled IN THE CAMERA
    PROCESS before it rides the Portal RPC (depth mode, 2026-08-05).

    Raw 640x480 uint16 depth is ~614 KB/frame/camera at 30 Hz — the same
    payload class that collapsed the control loop to 9-17 Hz when 448-px RGB
    shipped every tick (see launch.py's SubRL preprocess). Downscaling to the
    rgb_hd geometry (aspect-preserving, max side ``depth_max_px``) cuts the
    camera->main hop ~4x; launch then forwards depth main->agent only on the
    HD ticks. NEAREST interpolation: depth values must never be blended
    (bilinear invents surfaces between foreground and background edges).
    """

    depth_max_px: int = 448

    def read(self):
        data = super().read()
        depth = data.images.get("depth") if isinstance(data.images, dict) else None
        if depth is not None:
            h, w = depth.shape[:2]
            m = max(h, w)
            if m > self.depth_max_px:
                import cv2
                s = self.depth_max_px / float(m)
                small = cv2.resize(depth, (int(round(w * s)), int(round(h * s))),
                                   interpolation=cv2.INTER_NEAREST)
                small = np.ascontiguousarray(small)
                data.images["depth"] = small
            # Kill the depth_data ALIAS unconditionally (review 2026-08-05, CRITICAL):
            # robocam duplicates the frame into CameraData.depth_data, CameraNode
            # forwards it as a TOP-LEVEL obs key that lands in CameraObservation.extra
            # and re-emits on every Portal hop — bypassing the HD-tick gate (launch
            # deletes only images["depth"]). Nothing in limb consumes depth_data;
            # images["depth"] is the single carrier.
            data.depth_data = None
        return data
