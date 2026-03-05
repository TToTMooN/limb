"""Hands-free trigger signals for data collection.

Provides a unified interface for episode start/stop/discard signals
from various input devices. Essential for bimanual teleop where both
hands are occupied.

Supported backends:
  - Keyboard: spacebar/enter/escape (when a helper or one hand is free)
  - Foot pedal: USB HID pedals that present as keyboard (most practical)
  - VR buttons: B/Y buttons on Pico controllers (during VR teleop)

All backends implement the same TriggerSource protocol and are polled
via get_signal() → TriggerSignal | None.
"""

from __future__ import annotations

import enum
import select
import sys
import termios
import tty
from dataclasses import dataclass
from typing import Optional, Protocol


class TriggerSignal(enum.Enum):
    """Signals the operator can send during data collection."""

    START_STOP = "start_stop"  # Toggle recording on/off (primary action)
    DISCARD = "discard"  # Discard current episode
    SUCCESS = "success"  # Mark episode as success and save
    QUIT = "quit"  # End collection session


class TriggerSource(Protocol):
    """Protocol for reading operator signals."""

    def get_signal(self) -> Optional[TriggerSignal]:
        """Non-blocking poll for a signal. Returns None if no signal."""
        ...

    def close(self) -> None: ...


@dataclass
class KeyboardTrigger:
    """Read signals from keyboard (stdin).

    Key mapping:
      Space / Enter  →  START_STOP
      d              →  DISCARD
      s              →  SUCCESS (mark success and save)
      q / Escape     →  QUIT

    Works with USB foot pedals that present as keyboard HID devices
    (most foot pedals send Enter or Space by default).
    """

    def __post_init__(self) -> None:
        self._old_settings = None
        self._setup_terminal()

    def _setup_terminal(self) -> None:
        """Set terminal to raw mode for non-blocking key reads."""
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, OSError):
            self._old_settings = None

    def get_signal(self) -> Optional[TriggerSignal]:
        if not sys.stdin.isatty():
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if ch in (" ", "\r", "\n"):
            return TriggerSignal.START_STOP
        if ch == "d":
            return TriggerSignal.DISCARD
        if ch == "s":
            return TriggerSignal.SUCCESS
        if ch in ("q", "\x1b"):
            return TriggerSignal.QUIT
        return None

    def close(self) -> None:
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except (termios.error, OSError):
                pass


@dataclass
class VRButtonTrigger:
    """Read signals from VR controller buttons.

    Button mapping (uses the free B/Y buttons):
      B (right controller) →  START_STOP (toggle recording)
      Y (left controller)  →  DISCARD (discard current episode)

    Note: A and X are already used for arm reset in yam_vr_agent.py.

    Parameters
    ----------
    xr_client : XrClient instance from the VR agent.
    debounce_s : Minimum time between signals to avoid double-triggers.
    """

    xr_client: object = None  # XrClient, typed as object for lazy import
    debounce_s: float = 0.3

    def __post_init__(self) -> None:
        self._last_signal_time = 0.0
        self._prev_b = False
        self._prev_y = False

    def get_signal(self) -> Optional[TriggerSignal]:
        import time

        now = time.time()
        if now - self._last_signal_time < self.debounce_s:
            return None

        b_pressed = self.xr_client.get_button("B")
        y_pressed = self.xr_client.get_button("Y")

        signal = None
        # Detect rising edge (button just pressed)
        if b_pressed and not self._prev_b:
            signal = TriggerSignal.START_STOP
        elif y_pressed and not self._prev_y:
            signal = TriggerSignal.DISCARD

        self._prev_b = b_pressed
        self._prev_y = y_pressed

        if signal is not None:
            self._last_signal_time = now
        return signal

    def close(self) -> None:
        pass


@dataclass
class CompositeTrigger:
    """Combines multiple trigger sources. First signal wins.

    Usage in YAML:
        _target_: limb.recording.trigger.CompositeTrigger
        sources:
          - _target_: limb.recording.trigger.KeyboardTrigger
          - _target_: limb.recording.trigger.VRButtonTrigger
            xr_client: ...
    """

    sources: list = None

    def __post_init__(self) -> None:
        if self.sources is None:
            self.sources = []

    def get_signal(self) -> Optional[TriggerSignal]:
        for source in self.sources:
            signal = source.get_signal()
            if signal is not None:
                return signal
        return None

    def close(self) -> None:
        for source in self.sources:
            source.close()
