"""Core camera protocol and data types — zero framework dependencies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np


@dataclass
class IMUData:
    """Inertial measurement data from a camera or attached IMU."""

    timestamp: float  # relative timestamp in ms
    acceleration: Optional[Tuple[float, float, float]] = None  # [x, y, z]
    gyroscope: Optional[Tuple[float, float, float]] = None  # [x, y, z]


@dataclass
class CameraSpec:
    """Named shape/dtype descriptor for a camera stream."""

    name: str
    shape: Tuple[int, int, int]  # (height, width, channels)
    dtype: np.dtype


@dataclass
class CameraData:
    """Single capture from a camera driver."""

    images: Dict[str, np.ndarray]  # e.g. {"rgb": ..., "depth": ...}
    timestamp: float  # milliseconds
    calibration_data: Optional[dict] = None
    imu_data: Optional[IMUData] = None
    other_sensors: Optional[dict] = None


class CameraDriver(Protocol):
    """Protocol that all camera drivers must satisfy."""

    def read(self) -> CameraData:
        """Capture one frame (blocking)."""
        ...

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        """Return calibration intrinsics (K matrix, distortion, etc.)."""
        ...

    def get_camera_info(self) -> Dict[str, Any]:
        """Return device metadata (serial, resolution, fps, ...)."""
        ...

    def stop(self) -> None:
        """Release hardware resources."""
        ...


@dataclass
class DummyCamera:
    """Synthetic camera for testing — generates random noise images."""

    name: Optional[str] = None
    camera_specs: Optional[List[CameraSpec]] = None

    def __post_init__(self) -> None:
        if self.camera_specs is None:
            self.camera_specs = [
                CameraSpec(name=f"dummy_{i}", shape=(480, 640, 3), dtype=np.dtype(np.uint8)) for i in range(2)
            ]

    def __repr__(self) -> str:
        return f"DummyCamera({self.name})"

    def read(self) -> CameraData:
        assert self.camera_specs is not None
        images = {}
        for spec in self.camera_specs:
            images[spec.name] = np.random.randint(0, 255, spec.shape, dtype=np.uint8)
        return CameraData(images=images, timestamp=time.time() * 1000)

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        assert self.camera_specs is not None
        return {spec.name: {"K": np.eye(3), "D": np.random.rand(5)} for spec in self.camera_specs}

    def get_camera_info(self) -> Dict[str, Any]:
        return {"device_id": "dummy", "width": 640, "height": 480, "fps": 30}

    def stop(self) -> None:
        pass
