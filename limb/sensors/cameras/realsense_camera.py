import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as _e:
    raise ImportError(
        "pyrealsense2 is required for RealsenseCamera. "
        "Install it with:  pip install pyrealsense2"
    ) from _e

from limb.sensors.cameras.camera import CameraData, CameraDriver

logger = logging.getLogger(__name__)


def discover_devices() -> List[Dict[str, str]]:
    """Return ``[{serial, name}, ...]`` for every connected RealSense device."""
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        devices.append(
            {
                "serial": dev.get_info(rs.camera_info.serial_number),
                "name": dev.get_info(rs.camera_info.name),
            }
        )
    return devices


@dataclass
class RealsenseCamera(CameraDriver):
    """RealSense camera driver using the ``pyrealsense2`` SDK.

    Mirrors the interface of :class:`OpencvCamera` so it can be used as a
    drop-in replacement in YAML configs::

        camera:
            _target_: limb.sensors.cameras.realsense_camera.RealsenseCamera
            serial_number: "346123070863"
            enable_depth: true

    Discover connected cameras with::

        python -m limb.sensors.cameras.realsense_camera
        # or
        RealsenseCamera.discover_devices()
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
        logger.info("Opened RealSense %s (%s)", actual_serial, device.get_info(rs.camera_info.name))

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
        # CameraNode reads depth via getattr(data, "depth_data", None)
        data.depth_data = depth_array  # type: ignore[attr-defined]
        return data

    def get_camera_info(self) -> Dict[str, Any]:
        device = self.profile.get_device()
        info: Dict[str, Any] = {
            "camera_type": self.camera_type,
            "serial_number": device.get_info(rs.camera_info.serial_number),
            "name": device.get_info(rs.camera_info.name),
            "firmware_version": device.get_info(rs.camera_info.firmware_version),
            "width": self.resolution[0],
            "height": self.resolution[1],
            "fps": self.fps,
            "enable_depth": self.enable_depth,
        }
        return info

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        K = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ]
        )
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
            result["depth_K"] = np.array(
                [
                    [d_intr.fx, 0.0, d_intr.ppx],
                    [0.0, d_intr.fy, d_intr.ppy],
                    [0.0, 0.0, 1.0],
                ]
            )
            result["depth_D"] = np.array(d_intr.coeffs)
        return result

    def stop(self) -> None:
        self.pipeline.stop()
        logger.info("Stopped RealSense %s", self.serial_number)

    @staticmethod
    def discover_devices() -> List[Dict[str, str]]:
        """Return ``[{serial, name}, ...]`` for every connected RealSense device."""
        return discover_devices()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test a single RealSense camera")
    parser.add_argument("--serial", type=str, default=None, help="Serial number (omit for first available)")
    parser.add_argument("--list", action="store_true", help="List connected devices and exit")
    args = parser.parse_args()

    if args.list:
        devices = discover_devices()
        if not devices:
            print("No RealSense devices found.")
        for d in devices:
            print(f"  {d['serial']}  {d['name']}")
        raise SystemExit(0)

    camera = RealsenseCamera(serial_number=args.serial)
    print(f"Opened: {camera}")
    print(f"Info:   {camera.get_camera_info()}")
    try:
        while True:
            data = camera.read()
            print(f"ts={data.timestamp:.1f}ms  images={list(data.images.keys())}  "
                  f"shape={data.images['rgb'].shape}")
            time.sleep(1 / camera.fps)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
