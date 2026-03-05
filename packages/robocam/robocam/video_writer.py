"""Async video writer using ffmpeg subprocess with NVENC hardware acceleration.

Frames are piped to an ``ffmpeg`` child process in a background thread,
so ``write()`` never blocks the control loop for encoding. Falls back to
software encoding when NVENC is unavailable.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

_SENTINEL = None  # signals the writer thread to stop


def _check_nvenc_available() -> bool:
    """Probe whether the system has a working hevc_nvenc encoder."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "hevc_nvenc" in result.stdout
    except Exception:
        return False


# Cache the probe result per-process
_NVENC_AVAILABLE: Optional[bool] = None


def nvenc_available() -> bool:
    """Return True if hevc_nvenc is available (cached after first call)."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = _check_nvenc_available()
    return _NVENC_AVAILABLE


@dataclass
class AsyncVideoWriter:
    """Non-blocking video writer that encodes via ffmpeg in a background thread.

    Parameters
    ----------
    path : str
        Output ``.mp4`` file path.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    fps : int
        Output video frame rate.
    codec : str
        Preferred codec. ``"auto"`` selects ``hevc_nvenc`` if available, else ``libx264``.
        Other valid values: ``"hevc_nvenc"``, ``"h264_nvenc"``, ``"libx264"``, ``"libx265"``.
    pixel_format : str
        Output pixel format (default ``yuv420p`` for wide compatibility).
    crf : int
        Constant rate factor for quality (lower = better, only used with software codecs).
    queue_size : int
        Max frames buffered in memory before ``write()`` blocks. 0 = unlimited.

    Example
    -------
    >>> writer = AsyncVideoWriter("out.mp4", 640, 480, fps=30)
    >>> writer.start()
    >>> for frame in frames:
    ...     writer.write(frame)  # non-blocking (unless queue full)
    >>> writer.stop()            # flushes remaining frames
    """

    path: str
    width: int
    height: int
    fps: int = 30
    codec: str = "auto"
    pixel_format: str = "yuv420p"
    crf: int = 23
    queue_size: int = 300

    _proc: Optional[subprocess.Popen] = field(init=False, repr=False, default=None)
    _thread: Optional[threading.Thread] = field(init=False, repr=False, default=None)
    _queue: queue.Queue = field(init=False, repr=False)
    _frame_count: int = field(init=False, repr=False, default=0)
    _started: bool = field(init=False, repr=False, default=False)
    _failed: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=self.queue_size)

    def _resolve_codec(self) -> str:
        if self.codec != "auto":
            return self.codec
        return "hevc_nvenc" if nvenc_available() else "libx264"

    def _build_ffmpeg_cmd(self, codec: str) -> list[str]:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-i",
            "-",  # stdin
            "-an",  # no audio
            "-vcodec",
            codec,
            "-pix_fmt",
            self.pixel_format,
        ]
        # Quality settings differ between HW and SW codecs
        if codec in ("hevc_nvenc", "h264_nvenc"):
            cmd += ["-preset", "p4", "-rc", "constqp", "-qp", str(self.crf)]
        else:
            cmd += ["-crf", str(self.crf), "-preset", "fast"]
        cmd.append(self.path)
        return cmd

    def start(self) -> None:
        """Launch the ffmpeg subprocess and background writer thread."""
        if self._started:
            raise RuntimeError("Writer already started")

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        codec = self._resolve_codec()
        cmd = self._build_ffmpeg_cmd(codec)
        logger.info("AsyncVideoWriter: {} (codec={})", self.path, codec)

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True, name=f"video-writer-{Path(self.path).stem}"
        )
        self._thread.start()
        self._started = True

    def _writer_loop(self) -> None:
        """Drain the queue and pipe frames to ffmpeg stdin."""
        assert self._proc is not None and self._proc.stdin is not None
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                try:
                    self._proc.stdin.write(item)
                except BrokenPipeError:
                    logger.error("ffmpeg pipe broken — encoder may have crashed")
                    self._failed = True
                    break
        finally:
            try:
                self._proc.stdin.close()
            except Exception:
                pass

    def write(self, frame: np.ndarray) -> None:
        """Enqueue an RGB frame for encoding. Non-blocking unless queue is full.

        Parameters
        ----------
        frame : np.ndarray
            RGB uint8 image with shape ``(height, width, 3)``.
        """
        if not self._started:
            raise RuntimeError("Call start() before write()")
        if self._failed:
            return
        self._queue.put(frame.tobytes())
        self._frame_count += 1

    def stop(self) -> None:
        """Flush remaining frames and wait for ffmpeg to finish."""
        if not self._started:
            return

        if not self._failed:
            self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=10)

        if self._proc is not None:
            self._proc.wait(timeout=30)
            stderr = self._proc.stderr.read().decode() if self._proc.stderr else ""
            if self._proc.returncode != 0:
                logger.warning(
                    "ffmpeg exited with code {}: {}", self._proc.returncode, stderr[-500:] if stderr else ""
                )

        self._started = False
        logger.info("AsyncVideoWriter finished: {} ({} frames)", self.path, self._frame_count)

    @property
    def frame_count(self) -> int:
        """Number of frames submitted so far."""
        return self._frame_count

    def __del__(self) -> None:
        if self._started:
            try:
                self.stop()
            except Exception:
                pass
