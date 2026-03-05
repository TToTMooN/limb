"""Thread-safe frame history buffer for observation stacking.

Provides ``get_last_k(k)`` for VLA policies that need temporal context,
and ``get_latest()`` for real-time control loops that only need the most recent frame.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List

from robocam.camera import CameraData


@dataclass
class FrameBuffer:
    """Fixed-capacity, thread-safe ring buffer of :class:`CameraData` frames.

    Parameters
    ----------
    max_size : int
        Maximum number of frames to retain. When full, the oldest frame is dropped.
        Should be at least as large as the largest ``k`` you plan to request.

    Example
    -------
    >>> buf = FrameBuffer(max_size=10)
    >>> buf.put(camera.read())          # called from polling thread
    >>> latest = buf.get_latest()       # latest single frame
    >>> history = buf.get_last_k(4)     # last 4 frames, oldest first
    """

    max_size: int = 64

    _buf: deque = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)
    _frame_count: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._buf = deque(maxlen=self.max_size)
        self._lock = threading.Lock()
        self._frame_count = 0

    def put(self, data: CameraData) -> None:
        """Append a frame. Thread-safe, non-blocking."""
        with self._lock:
            self._buf.append(data)
            self._frame_count += 1

    def get_latest(self, timeout_sec: float = 1.0) -> CameraData:
        """Return the most recent frame.

        Raises
        ------
        TimeoutError
            If the buffer is empty and no frame arrives within *timeout_sec*.
        """
        deadline = time.monotonic() + timeout_sec
        while True:
            with self._lock:
                if self._buf:
                    return self._buf[-1]
            if time.monotonic() >= deadline:
                raise TimeoutError(f"No frame received within {timeout_sec}s")
            time.sleep(0.001)

    def get_last_k(self, k: int) -> List[CameraData]:
        """Return up to the last *k* frames, ordered oldest-first.

        If fewer than *k* frames are available, returns all available frames.
        Returns an empty list if the buffer is empty.
        """
        with self._lock:
            n = len(self._buf)
            start = max(0, n - k)
            return list(self._buf)[start:]

    @property
    def count(self) -> int:
        """Total number of frames ever inserted (monotonically increasing)."""
        return self._frame_count

    def __len__(self) -> int:
        """Number of frames currently in the buffer."""
        with self._lock:
            return len(self._buf)

    def clear(self) -> None:
        """Drop all buffered frames."""
        with self._lock:
            self._buf.clear()
