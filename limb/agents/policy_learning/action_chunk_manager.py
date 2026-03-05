"""Action chunk buffering and temporal smoothing for policy agents.

When a new chunk arrives from inference, it is blended with the remaining
actions in the old buffer to avoid discontinuities. Thread-safe so
inference can run asynchronously.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class ActionChunkManager:
    """Manages an action chunk buffer with temporal smoothing across chunks.

    Parameters
    ----------
    action_horizon : int
        Number of timesteps in each action chunk.
    smoothing_window : int
        Number of overlapping actions to blend between old and new chunks.
        0 = no smoothing (hard switch).
    """

    action_horizon: int = 25
    smoothing_window: int = 4

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: np.ndarray | None = None
        self._index: int = 0

    def update(self, new_chunk: np.ndarray, steps_since_request: int = 0) -> None:
        """Receive a new action chunk from inference.

        Args:
            new_chunk: (action_horizon, action_dim) array.
            steps_since_request: Control steps elapsed since the inference
                request was sent (for latency compensation).
        """
        with self._lock:
            offset = min(steps_since_request, new_chunk.shape[0] - 1)
            new_chunk = new_chunk[offset:]

            if self._buffer is None:
                self._buffer = new_chunk
                self._index = 0
                return

            remaining = self._buffer[self._index :]
            n_smooth = min(self.smoothing_window, remaining.shape[0], new_chunk.shape[0])

            if n_smooth > 0:
                weights = np.linspace(1.0 / n_smooth, 1.0, n_smooth).reshape(-1, 1)
                blended = weights * new_chunk[:n_smooth] + (1 - weights) * remaining[:n_smooth]
                self._buffer = np.concatenate([blended, new_chunk[n_smooth:]], axis=0)
            else:
                self._buffer = new_chunk
            self._index = 0

    def get_action(self) -> np.ndarray | None:
        """Pop the next action from the buffer.

        Returns:
            1-D action array, or None if no actions are available.
        """
        with self._lock:
            if self._buffer is None:
                return None
            if self._index >= self._buffer.shape[0]:
                return self._buffer[-1]  # repeat last action if exhausted
            action = self._buffer[self._index]
            self._index += 1
            return action

    @property
    def has_actions(self) -> bool:
        with self._lock:
            return self._buffer is not None

    @property
    def remaining(self) -> int:
        with self._lock:
            if self._buffer is None:
                return 0
            return max(0, self._buffer.shape[0] - self._index)
