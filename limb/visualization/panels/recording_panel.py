"""Recording panel — simple Start/Stop recording button in the viser sidebar.

Used in non-session mode (standalone teleop). For data collection sessions,
use SessionPanel instead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import viser
from loguru import logger


@dataclass
class RecordingPanel:
    """One-button video recording toggle in the viser GUI.

    Requires a callback to get current RGB frames (typically from CameraPanel).

    Implements ViserPanel: attach(server), detach().
    """

    recording_fps: int = 30
    get_latest_rgb: Optional[Callable[[], Dict[str, Any]]] = None

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

    def update_frame(self, rgb_images: Dict[str, Any]) -> None:
        """Write current frames to video if recording. Call from control loop."""
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
        with self._lock:
            if not self._recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._record_dir = Path("recordings") / f"recording_{ts}"
                self._record_dir.mkdir(parents=True, exist_ok=True)

                # Pre-create writers from latest frames if available
                if self.get_latest_rgb is not None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    for cam_name, img in self.get_latest_rgb().items():
                        h, w = img.shape[:2]
                        path = str(self._record_dir / f"{cam_name}.mp4")
                        self._writers[cam_name] = cv2.VideoWriter(path, fourcc, self.recording_fps, (w, h))

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
        with self._lock:
            if self._recording:
                for w in self._writers.values():
                    w.release()
                logger.info("Recording saved to {}", self._record_dir)
                self._writers.clear()
                self._recording = False
