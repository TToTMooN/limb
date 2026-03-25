"""Recording panel — simple Start/Stop recording button in the viser sidebar.

Used in non-session mode (standalone teleop). For data collection sessions,
use SessionPanel instead.

Threading model:
- ``update(obs)`` is called from the **main thread** (100 Hz control loop).
  It writes video frames while holding ``_lock``.
- ``_toggle()`` is called from the **viser server thread** (on_click callback).
  It also holds ``_lock`` to safely start/stop recording.
- ``get_latest_rgb`` callback reads from CameraPanel which is updated on the
  main thread before this panel (registration order guarantees this).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import viser
from loguru import logger

from limb.sensors.cameras.camera_utils import obs_get_rgb


@dataclass
class RecordingPanel:
    """One-button video recording toggle in the viser GUI.

    Implements ObservablePanel: attach(server), update(obs), detach().

    The ``update(obs)`` method extracts RGB from the observation and writes
    frames to video when recording is active. This is called from the main
    thread, after CameraPanel has already processed the same observation.
    """

    recording_fps: int = 30

    _server: Optional[viser.ViserServer] = field(default=None, init=False, repr=False)
    _button: Optional[Any] = field(default=None, init=False, repr=False)
    _recording: bool = field(default=False, init=False, repr=False)
    _writers: Dict[str, cv2.VideoWriter] = field(default_factory=dict, init=False, repr=False)
    _record_dir: Optional[Path] = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def attach(self, server: viser.ViserServer) -> None:
        self._server = server
        self._button = server.gui.add_button("Start Recording", color="green")

        @self._button.on_click
        def _(_event: viser.GuiEvent) -> None:
            self._toggle()

    def update(self, obs: Any) -> None:
        """Extract RGB from obs and write frames if recording.

        Called from main thread at control loop rate.
        """
        with self._lock:
            if not self._recording:
                return

        # Extract outside lock — obs_get_rgb is read-only
        rgb_images = obs_get_rgb(obs)
        if not rgb_images:
            return

        with self._lock:
            if not self._recording:
                return
            for cam_name, img in rgb_images.items():
                writer = self._writers.get(cam_name)
                if writer is None:
                    h, w = img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    path = str(self._record_dir / f"{cam_name}.mp4")
                    writer = cv2.VideoWriter(path, fourcc, self.recording_fps, (w, h))
                    self._writers[cam_name] = writer
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _toggle(self) -> None:
        """Toggle recording on/off. Called from viser server thread."""
        with self._lock:
            if not self._recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._record_dir = Path("recordings") / f"recording_{ts}"
                self._record_dir.mkdir(parents=True, exist_ok=True)

                self._recording = True
                self._button.name = "Stop Recording"
                self._button.color = "red"
                logger.info("Recording started -> {}", self._record_dir)
            else:
                for w in self._writers.values():
                    w.release()
                logger.info("Recording saved to {}", self._record_dir)
                self._writers.clear()
                self._record_dir = None
                self._recording = False
                self._button.name = "Start Recording"
                self._button.color = "green"

    def detach(self) -> None:
        """Stop recording and release all writers."""
        with self._lock:
            if self._recording:
                for w in self._writers.values():
                    w.release()
                logger.info("Recording saved to {}", self._record_dir)
                self._writers.clear()
                self._recording = False
