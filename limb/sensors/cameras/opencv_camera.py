import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from limb.sensors.cameras.camera import CameraData, CameraDriver

logger = logging.getLogger(__name__)

V4L_BY_ID_DIR = Path("/dev/v4l/by-id")


def resolve_device_by_serial(serial_number: str, video_index: int = 0) -> str:
    """Resolve a RealSense serial number to a ``/dev/videoX`` path.

    Scans ``/dev/v4l/by-id/`` for symlinks whose name contains the given
    *serial_number* and ends with ``-video-index{video_index}``.
    """
    if not V4L_BY_ID_DIR.is_dir():
        raise FileNotFoundError(
            f"{V4L_BY_ID_DIR} does not exist — is the camera plugged in?"
        )

    suffix = f"-video-index{video_index}"
    for entry in V4L_BY_ID_DIR.iterdir():
        if serial_number in entry.name and entry.name.endswith(suffix):
            resolved = str(entry.resolve())
            logger.info("Resolved serial %s (index %d) → %s", serial_number, video_index, resolved)
            return resolved

    available = [e.name for e in V4L_BY_ID_DIR.iterdir()]
    raise FileNotFoundError(
        f"No V4L device found for serial '{serial_number}' with video-index{video_index}. "
        f"Available entries: {available}"
    )


@dataclass
class OpencvCamera(CameraDriver):
    """OpenCV-based camera driver with optional serial-number resolution.

    If *serial_number* is set, the device path is resolved automatically from
    ``/dev/v4l/by-id/`` so the config is stable across reboots/replugs.
    Otherwise *device_path* is used directly (legacy behaviour).

    Discover serials with::

        ls /dev/v4l/by-id/
        # or
        v4l2-ctl --list-devices
    """

    device_path: str = ""
    serial_number: Optional[str] = None
    video_index: int = 0
    camera_type: str = "zed_camera"
    image_transfer_time_offset: int = 80  # ms typical transfer time
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    name: Optional[str] = None

    def __repr__(self) -> str:
        id_str = self.serial_number or self.device_path
        return f"OpencvCamera({id_str!r}, name={self.name!r}, resolution={self.resolution}, fps={self.fps})"

    def __post_init__(self):
        if self.serial_number:
            self.device_path = resolve_device_by_serial(self.serial_number, self.video_index)
        elif not self.device_path:
            raise ValueError("Either 'serial_number' or 'device_path' must be provided")

        logger.info("Opening camera at %s", self.device_path)
        self.cap = cv2.VideoCapture(self.device_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at {self.device_path}")

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> CameraData:
        try:
            ret, frame = self.cap.read()
            capture_time_ms = time.time() * 1000
            while not ret:
                # If read failed, retry and update capture time
                ret, frame = self.cap.read()
                capture_time_ms = time.time() * 1000
                time.sleep(0.01)
        except Exception as e:
            logging.error(f"Error reading frame: {e}")
            raise e
        frame = np.ascontiguousarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Adjust timestamp using the offset
        timestamp_ms = capture_time_ms - self.image_transfer_time_offset
        return CameraData(images={"rgb": frame}, timestamp=timestamp_ms)

    def get_camera_info(self) -> dict:
        info = {
            "camera_type": self.camera_type,
            "device_path": self.device_path,
            "width": self.resolution[0],
            "height": self.resolution[1],
            "fps": self.fps,
        }
        if self.serial_number:
            info["serial_number"] = self.serial_number
        return info

    def stop(self) -> None:
        self.cap.release()

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Calibration data reading is not implemented for {self}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--device_path", type=str, help="e.g. /dev/video0")
    group.add_argument("--serial", type=str, help="RealSense serial number")
    args = parser.parse_args()

    if args.serial:
        camera = OpencvCamera(serial_number=args.serial)
    else:
        camera = OpencvCamera(device_path=args.device_path)

    while True:
        data = camera.read()
        print(data)
        time.sleep(1 / camera.fps)
