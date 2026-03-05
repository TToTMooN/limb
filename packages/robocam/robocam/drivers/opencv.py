"""Generic V4L2 / OpenCV camera driver.

Requires: ``pip install robocam[opencv]``
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger

from robocam.camera import CameraData

try:
    import cv2
except ImportError as _e:
    raise ImportError("opencv-contrib-python is required: pip install robocam[opencv]") from _e

V4L_BY_ID_DIR = Path("/dev/v4l/by-id")


def resolve_device_by_serial(serial_number: str, video_index: int = 0) -> str:
    """Resolve a serial number to a ``/dev/videoX`` path via ``/dev/v4l/by-id/``."""
    if not V4L_BY_ID_DIR.is_dir():
        raise FileNotFoundError(f"{V4L_BY_ID_DIR} does not exist — is the camera plugged in?")

    suffix = f"-video-index{video_index}"
    for entry in V4L_BY_ID_DIR.iterdir():
        if serial_number in entry.name and entry.name.endswith(suffix):
            resolved = str(entry.resolve())
            logger.info("Resolved serial {} (index {}) -> {}", serial_number, video_index, resolved)
            return resolved

    available = [e.name for e in V4L_BY_ID_DIR.iterdir()]
    raise FileNotFoundError(
        f"No V4L device found for serial '{serial_number}' with video-index{video_index}. "
        f"Available entries: {available}"
    )


@dataclass
class OpencvCamera:
    """OpenCV-based camera driver with optional serial-number resolution.

    Parameters
    ----------
    device_path : str
        ``/dev/videoX`` path. Ignored if *serial_number* is set.
    serial_number : str or None
        If set, resolves the device path from ``/dev/v4l/by-id/``.
    video_index : int
        V4L video-index suffix used during serial resolution.
    image_transfer_time_offset : int
        Milliseconds to subtract from wall-clock time to approximate true capture time.
    resolution : tuple
        ``(width, height)`` requested from the driver.
    fps : int
        Target frame rate.
    name : str or None
        Human-readable label.
    """

    device_path: str = ""
    serial_number: Optional[str] = None
    video_index: int = 0
    camera_type: str = "opencv_camera"
    image_transfer_time_offset: int = 80  # ms
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    name: Optional[str] = None

    def __repr__(self) -> str:
        id_str = self.serial_number or self.device_path
        return f"OpencvCamera({id_str!r}, name={self.name!r}, resolution={self.resolution}, fps={self.fps})"

    def __post_init__(self) -> None:
        if self.serial_number:
            self.device_path = resolve_device_by_serial(self.serial_number, self.video_index)
        elif not self.device_path:
            raise ValueError("Either 'serial_number' or 'device_path' must be provided")

        logger.info("Opening camera at {}", self.device_path)
        self.cap = cv2.VideoCapture(self.device_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at {self.device_path}")

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> CameraData:
        ret, frame = self.cap.read()
        capture_time_ms = time.time() * 1000
        while not ret:
            ret, frame = self.cap.read()
            capture_time_ms = time.time() * 1000
            time.sleep(0.01)

        frame = np.ascontiguousarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return CameraData(images={"rgb": frame}, timestamp=capture_time_ms - self.image_transfer_time_offset)

    def get_camera_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "camera_type": self.camera_type,
            "device_path": self.device_path,
            "width": self.resolution[0],
            "height": self.resolution[1],
            "fps": self.fps,
        }
        if self.serial_number:
            info["serial_number"] = self.serial_number
        return info

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Calibration data reading is not implemented for {self}")

    def stop(self) -> None:
        self.cap.release()
