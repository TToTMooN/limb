"""robocam — Minimal camera drivers, async video encoding, and frame buffering for robotics."""

__version__ = "0.1.0"

from robocam.camera import CameraData, CameraDriver, CameraSpec, IMUData
from robocam.frame_buffer import FrameBuffer
from robocam.video_writer import AsyncVideoWriter

__all__ = [
    "CameraData",
    "CameraDriver",
    "CameraSpec",
    "IMUData",
    "FrameBuffer",
    "AsyncVideoWriter",
]
