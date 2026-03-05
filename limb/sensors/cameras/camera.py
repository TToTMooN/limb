"""Camera protocol, data types (from robocam), and CameraNode Portal wrapper."""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from loguru import logger
from robocam.camera import CameraData, CameraDriver, CameraSpec, DummyCamera, IMUData  # noqa: F401

from limb.utils.portal_utils import remote


@dataclass
class CameraNode:
    """Portal-RPC wrapper that polls a CameraDriver in a background thread.

    This is limb-specific glue — it adds Portal ``@remote()`` methods and
    a daemon polling thread on top of any :class:`robocam.camera.CameraDriver`.
    """

    camera: CameraDriver
    timeout_sec: float = 1.0

    def __post_init__(self) -> None:
        self.latest_data: Optional[CameraData] = None
        self.last_update_time: Optional[float] = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.polling_thread = threading.Thread(target=self._poll_image, daemon=True)
        self.polling_thread.start()
        self.exception: Optional[Exception] = None

    def _poll_image(self) -> None:
        while not self.stop_event.is_set():
            try:
                latest_data = self.camera.read()
                with self.lock:
                    self.latest_data = latest_data
                    self.last_update_time = time.time()
                time.sleep(0.004)
            except Exception as e:
                self.exception = e
                logger.error("Error polling camera {}: {}", self.camera, e)

    @remote()
    def read(self) -> Dict[str, Any]:
        if self.exception is not None:
            raise self.exception
        while self.latest_data is None:
            if not self.polling_thread.is_alive():
                raise RuntimeError("Polling thread died before first image was received")
            logger.debug("Waiting for first frame from {}", self.camera)
            time.sleep(0.1)

        with self.lock:
            if self.last_update_time is None:
                raise RuntimeError("No data received yet")
            if time.time() - self.last_update_time > self.timeout_sec:
                raise TimeoutError(f"No new camera data for {self.camera} in the last {self.timeout_sec} seconds")
            return self._get_latest_data()

    def _get_latest_data(self) -> Dict[str, Any]:
        assert self.latest_data is not None
        result = dict(images=self.latest_data.images, timestamp=self.latest_data.timestamp)
        depth_data = getattr(self.latest_data, "depth_data", None)
        if depth_data is not None:
            result["depth_data"] = depth_data
        intrinsic_data = getattr(self.camera, "intrinsic_data", None)
        if intrinsic_data is not None:
            result["intrinsics"] = self.camera.read_calibration_data_intrinsics()
        return result

    @remote(serialization_needed=True)
    def get_camera_info(self) -> Dict[str, Any]:
        return self.camera.get_camera_info()

    @remote()
    def close(self) -> None:
        self.stop_event.set()
        self.camera.stop()
        self.polling_thread.join()
