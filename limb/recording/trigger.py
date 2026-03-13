"""Hands-free trigger signals for data collection.

Provides a unified interface for episode start/stop/discard signals
from various input devices. Essential for bimanual teleop where both
hands are occupied.

Supported backends:
  - Keyboard: spacebar/enter/escape (when a helper or one hand is free)
  - Foot pedal: USB HID pedals via evdev with exclusive grab (iKKEGOL/PCsensor)
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
class FootPedalTrigger:
    """Read signals from a USB foot pedal via evdev (grabs device exclusively).

    Designed for iKKEGOL / PCsensor double foot pedals. The pedal must be
    programmed to send keyboard keys (factory default is typically A and B).

    Default key mapping (configurable):
      Left pedal  (KEY_A) →  START_STOP (toggle recording)
      Right pedal (KEY_B) →  DISCARD (discard current episode)

    The device is identified by matching vendor:product ID or /dev/input/by-id
    path. It is grabbed exclusively so key events don't leak to the desktop.

    Parameters
    ----------
    device_path : Path to the evdev device, or "auto" to find by vendor/product.
    vendor_id : USB vendor ID (hex). Default 0x3553 = PCsensor.
    product_id : USB product ID (hex). Default 0xb001 = FootSwitch.
    left_key : evdev key name for left pedal.
    right_key : evdev key name for right pedal.
    left_signal : Signal to emit for left pedal.
    right_signal : Signal to emit for right pedal.
    """

    device_path: str = "auto"
    vendor_id: int = 0x3553
    product_id: int = 0xB001
    left_key: str = "KEY_A"
    right_key: str = "KEY_B"
    left_signal: str = "START_STOP"
    right_signal: str = "DISCARD"

    def __post_init__(self) -> None:
        import evdev as _evdev
        from loguru import logger

        self._device: _evdev.InputDevice | None = None

        if self.device_path == "auto":
            self._device = self._find_device()
        else:
            self._device = _evdev.InputDevice(self.device_path)

        if self._device is None:
            logger.warning(
                f"Foot pedal not found (vendor={self.vendor_id:#06x}, product={self.product_id:#06x}). "
                "FootPedalTrigger will be inactive."
            )
            return

        self._device.grab()
        logger.info(f"Foot pedal grabbed: {self._device.name} ({self._device.path})")

        self._left_code = _evdev.ecodes.ecodes[self.left_key]
        self._right_code = _evdev.ecodes.ecodes[self.right_key]
        self._left_signal = TriggerSignal(self.left_signal.lower())
        self._right_signal = TriggerSignal(self.right_signal.lower())

    def _find_device(self) -> object | None:
        import evdev as _evdev

        candidates = []
        for path in _evdev.list_devices():
            try:
                dev = _evdev.InputDevice(path)
                info = dev.info
                if info.vendor == self.vendor_id and info.product == self.product_id:
                    if _evdev.ecodes.EV_KEY in dev.capabilities():
                        candidates.append(dev)
                    else:
                        dev.close()
                else:
                    dev.close()
            except (OSError, PermissionError):
                continue
        if not candidates:
            return None
        # Prefer the keyboard interface (name contains "Keyboard") over mouse
        for dev in candidates:
            if "keyboard" in dev.name.lower():
                for other in candidates:
                    if other is not dev:
                        other.close()
                return dev
        # Fallback to first candidate
        for dev in candidates[1:]:
            dev.close()
        return candidates[0]

    def get_signal(self) -> Optional[TriggerSignal]:
        if self._device is None:
            return None
        import select as _select

        r, _, _ = _select.select([self._device], [], [], 0)
        if not r:
            return None
        signal = None
        for event in self._device.read():
            if event.type == 1 and event.value == 1:  # EV_KEY, key down
                if event.code == self._left_code:
                    signal = self._left_signal
                elif event.code == self._right_code:
                    signal = self._right_signal
        return signal

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.ungrab()
            except OSError:
                pass
            self._device.close()
            self._device = None


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
