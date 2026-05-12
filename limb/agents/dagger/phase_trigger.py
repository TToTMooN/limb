"""Phase-trigger inputs for DAgger.

A :class:`PhaseTrigger` runs a background listener thread that pushes
``pause_resume`` / ``correction`` events into a :class:`DAggerEvents`
inbox owned by the :class:`DAggerAgent`.  The composite agent's per-tick
``consume_transition`` then turns those events into phase changes.

Currently only the foot-pedal trigger is implemented (matches the
production UX).  A keyboard trigger can be added later by following the
same Protocol.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from loguru import logger

from limb.agents.dagger.phase import DAggerEvents


class PhaseTrigger(Protocol):
    """Pushes phase-transition events into a DAgger inbox.

    Implementations spawn their own listener thread in ``start`` and stop
    it in ``close``.  They never block the control loop.
    """

    def start(self, events: DAggerEvents) -> None: ...
    def close(self) -> None: ...


@dataclass
class FootPedalPhaseTrigger:
    """USB foot pedal -> DAgger phase events.

    Designed for the same iKKEGOL/PCsensor pedals limb already uses for
    recording triggers (see :class:`limb.recording.trigger.FootPedalTrigger`).
    The pedal must be programmed to send keyboard keys; defaults match the
    factory programming on PCsensor double pedals.

    Bindings (configurable via YAML):

    * ``pause_resume_key`` (default ``KEY_A``) — toggle AUTONOMOUS<->PAUSED
    * ``correction_key``   (default ``KEY_B``) — toggle PAUSED<->CORRECTING

    The device is grabbed exclusively so key events don't leak to the
    desktop session.

    Parameters
    ----------
    device_path : str
        Either a concrete evdev path (``/dev/input/by-id/...``) or
        ``"auto"`` to auto-discover by USB vendor/product ID.
    vendor_id : int
        USB vendor ID for auto-discovery.  Default ``0x3553`` (PCsensor).
    product_id : int
        USB product ID for auto-discovery.  Default ``0xb001`` (FootSwitch).
    pause_resume_key, correction_key : str
        evdev key names assigned to each pedal.
    debounce_s : float
        Minimum seconds between accepted presses on the same key.
    """

    device_path: str = "auto"
    vendor_id: int = 0x3553
    product_id: int = 0xB001
    pause_resume_key: str = "KEY_A"
    correction_key: str = "KEY_B"
    debounce_s: float = 0.3

    _device: object = field(default=None, init=False, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self, events: DAggerEvents) -> None:
        import evdev as _evdev

        self._device = self._open_device(_evdev)
        if self._device is None:
            logger.warning(
                "FootPedalPhaseTrigger: pedal not found "
                f"(vendor={self.vendor_id:#06x}, product={self.product_id:#06x}). "
                "DAgger phase changes will not be available — keyboard ESC still stops the run."
            )
            return

        self._device.grab()
        logger.info(
            "FootPedalPhaseTrigger grabbed: {} ({}). pause_resume={} correction={}",
            self._device.name,
            self._device.path,
            self.pause_resume_key,
            self.correction_key,
        )

        pause_code = _evdev.ecodes.ecodes[self.pause_resume_key]
        correction_code = _evdev.ecodes.ecodes[self.correction_key]

        self._thread = threading.Thread(
            target=self._listen,
            args=(events, pause_code, correction_code),
            name="dagger-phase-pedal",
            daemon=True,
        )
        self._thread.start()

    def _open_device(self, _evdev):
        if self.device_path != "auto":
            return _evdev.InputDevice(self.device_path)

        candidates = []
        for path in _evdev.list_devices():
            try:
                dev = _evdev.InputDevice(path)
            except (OSError, PermissionError):
                continue
            try:
                info = dev.info
                if (
                    info.vendor == self.vendor_id
                    and info.product == self.product_id
                    and _evdev.ecodes.EV_KEY in dev.capabilities()
                ):
                    candidates.append(dev)
                else:
                    dev.close()
            except OSError:
                dev.close()

        if not candidates:
            return None
        # Prefer the keyboard interface over the mouse interface on combo devices.
        for dev in candidates:
            if "keyboard" in dev.name.lower():
                for other in candidates:
                    if other is not dev:
                        other.close()
                return dev
        for dev in candidates[1:]:
            dev.close()
        return candidates[0]

    def _listen(self, events: DAggerEvents, pause_code: int, correction_code: int) -> None:
        last_press: dict[int, float] = {}
        # Blocking read loop with a short timeout so close() can land cleanly.
        import select as _select

        while not self._stop.is_set():
            try:
                r, _, _ = _select.select([self._device], [], [], 0.2)
                if not r:
                    continue
                for event in self._device.read():
                    # EV_KEY = 1, value 1 = key down (ignore repeats / releases)
                    if event.type != 1 or event.value != 1:
                        continue
                    now = time.monotonic()
                    if now - last_press.get(event.code, 0.0) < self.debounce_s:
                        continue
                    last_press[event.code] = now
                    if event.code == pause_code:
                        accepted = events.request_transition("pause_resume")
                        logger.debug("pedal pause_resume accepted={}", accepted)
                    elif event.code == correction_code:
                        accepted = events.request_transition("correction")
                        logger.debug("pedal correction accepted={}", accepted)
            except OSError as e:
                logger.warning("FootPedalPhaseTrigger read error: {}", e)
                self._stop.wait(0.5)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._device is not None:
            try:
                self._device.ungrab()
            except OSError:
                pass
            self._device.close()
            self._device = None
