"""Intel RealSense D400-series camera driver.

Requires: ``pip install robocam[realsense]``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from robocam.camera import CameraData

try:
    import pyrealsense2 as rs
except ImportError as _e:
    raise ImportError("pyrealsense2 is required: pip install robocam[realsense]") from _e


def discover_devices() -> List[Dict[str, str]]:
    """Return ``[{serial, name}, ...]`` for every connected RealSense device."""
    ctx = rs.context()
    return [
        {
            "serial": dev.get_info(rs.camera_info.serial_number),
            "name": dev.get_info(rs.camera_info.name),
        }
        for dev in ctx.query_devices()
    ]


@dataclass
class RealsenseCamera:
    """RealSense camera driver using the ``pyrealsense2`` SDK.

    Parameters
    ----------
    serial_number : str or None
        Device serial. ``None`` picks the first available camera.
    resolution : tuple
        ``(width, height)`` of the color stream.
    fps : int
        Target frame rate.
    enable_depth : bool
        Enable aligned depth stream alongside color.
    name : str or None
        Human-readable label (stored in metadata, not used by SDK).
    """

    serial_number: Optional[str] = None
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    enable_depth: bool = False
    camera_type: str = "realsense_camera"
    name: Optional[str] = None

    pipeline: Any = field(init=False, repr=False)
    profile: Any = field(init=False, repr=False)
    _align: Any = field(init=False, repr=False, default=None)

    def __repr__(self) -> str:
        id_str = self.serial_number or "first-available"
        return f"RealsenseCamera({id_str!r}, name={self.name!r}, resolution={self.resolution}, fps={self.fps})"

    def __post_init__(self) -> None:
        cfg = rs.config()
        if self.serial_number:
            cfg.enable_device(self.serial_number)

        w, h = self.resolution
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, self.fps)
        if self.enable_depth:
            cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, self.fps)

        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(cfg)

        if self.enable_depth:
            self._align = rs.align(rs.stream.color)

        device = self.profile.get_device()
        actual_serial = device.get_info(rs.camera_info.serial_number)
        if self.serial_number is None:
            self.serial_number = actual_serial
        logger.info("Opened RealSense {} ({})", actual_serial, device.get_info(rs.camera_info.name))

    def read(self) -> CameraData:
        frames = self.pipeline.wait_for_frames()

        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to get color frame from RealSense pipeline")

        capture_time_ms = time.time() * 1000
        color_image = np.ascontiguousarray(np.asarray(color_frame.get_data()))

        images: Dict[str, np.ndarray] = {"rgb": color_image}
        depth_array: Optional[np.ndarray] = None

        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_array = np.ascontiguousarray(np.asarray(depth_frame.get_data()))
                images["depth"] = depth_array

        data = CameraData(images=images, timestamp=capture_time_ms)
        data.depth_data = depth_array  # type: ignore[attr-defined]
        return data

    def get_camera_info(self) -> Dict[str, Any]:
        device = self.profile.get_device()
        return {
            "camera_type": self.camera_type,
            "serial_number": device.get_info(rs.camera_info.serial_number),
            "name": device.get_info(rs.camera_info.name),
            "firmware_version": device.get_info(rs.camera_info.firmware_version),
            "width": self.resolution[0],
            "height": self.resolution[1],
            "fps": self.fps,
            "enable_depth": self.enable_depth,
        }

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
        D = np.array(intr.coeffs)
        result: Dict[str, Any] = {
            "K": K,
            "D": D,
            "width": intr.width,
            "height": intr.height,
            "model": str(intr.model),
        }
        if self.enable_depth:
            depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            d_intr = depth_stream.get_intrinsics()
            result["depth_K"] = np.array([[d_intr.fx, 0.0, d_intr.ppx], [0.0, d_intr.fy, d_intr.ppy], [0.0, 0.0, 1.0]])
            result["depth_D"] = np.array(d_intr.coeffs)
        return result

    def stop(self) -> None:
        self.pipeline.stop()
        logger.info("Stopped RealSense {}", self.serial_number)

    @staticmethod
    def discover_devices() -> List[Dict[str, str]]:
        """Return ``[{serial, name}, ...]`` for every connected RealSense device."""
        return discover_devices()
