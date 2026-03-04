"""Joy-Con gripper reader for bimanual teleoperation.

Reads Nintendo Joy-Con (L + R) controllers via ``evdev`` and maps their
inputs to gripper values in the range ``[gripper_open, gripper_close]``.

Two control modes work simultaneously:

* **Proportional** -- joystick Y-axis deflection controls the *velocity*
  of the gripper (push forward to close, pull back to open).
* **Toggle** -- ZL (left) / ZR (right) button press snaps the gripper
  between fully open and fully closed.

Each Joy-Con is handled by its own background thread with automatic
device detection and reconnection.
"""

import logging
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

LEFT_JOYCON_PRODUCT_ID = 0x2006
RIGHT_JOYCON_PRODUCT_ID = 0x2007
JOYCON_VENDOR_ID = 0x057E

RECONNECT_INTERVAL = 2.0


def _find_joycon(side: str) -> "Optional[str]":
    """Return the evdev device path for the requested Joy-Con side, or None."""
    from evdev import InputDevice, list_devices

    target_product = LEFT_JOYCON_PRODUCT_ID if side == "left" else RIGHT_JOYCON_PRODUCT_ID
    for path in list_devices():
        try:
            dev = InputDevice(path)
            name = dev.name.lower()
            if (
                dev.info.vendor == JOYCON_VENDOR_ID
                and dev.info.product == target_product
                and "imu" not in name
                and "motion" not in name
            ):
                return path
        except Exception:
            continue
    return None


class JoyConGripperReader:
    """Thread-safe reader that turns Joy-Con inputs into gripper values.

    Parameters
    ----------
    gripper_open : float
        Value representing fully open gripper (default 0.0).
    gripper_close : float
        Value representing fully closed gripper (default 2.4).
    stick_speed : float
        Gripper velocity (units/s) at full stick deflection.
    deadzone : float
        Stick axis values below this threshold are treated as zero.
    """

    def __init__(
        self,
        gripper_open: float = 0.0,
        gripper_close: float = 2.4,
        stick_speed: float = 3.0,
        deadzone: float = 0.05,
    ) -> None:
        self._open = gripper_open
        self._close = gripper_close
        self._lo = min(gripper_open, gripper_close)
        self._hi = max(gripper_open, gripper_close)
        self._stick_speed = stick_speed
        self._deadzone = deadzone

        self._lock = threading.Lock()
        self._left_gripper = gripper_open
        self._right_gripper = gripper_open
        self._stop = threading.Event()

        self._left_thread = threading.Thread(target=self._run, args=("left",), daemon=True)
        self._right_thread = threading.Thread(target=self._run, args=("right",), daemon=True)
        self._left_thread.start()
        self._right_thread.start()
        logger.info(
            "JoyConGripperReader started (range %.1f–%.1f, speed %.1f/s)",
            self._lo,
            self._hi,
            self._stick_speed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_gripper_values(self) -> Tuple[float, float]:
        """Return ``(left_gripper, right_gripper)``."""
        with self._lock:
            return self._left_gripper, self._right_gripper

    def close(self) -> None:
        self._stop.set()
        self._left_thread.join(timeout=3)
        self._right_thread.join(timeout=3)
        logger.info("JoyConGripperReader closed")

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self, side: str) -> None:
        """Event loop for one Joy-Con (left or right)."""
        import select

        from evdev import InputDevice, ecodes

        if side == "left":
            axis_code = ecodes.ABS_Y
            toggle_code = ecodes.BTN_TL2  # ZL
        else:
            axis_code = ecodes.ABS_RY
            toggle_code = ecodes.BTN_TR2  # ZR

        while not self._stop.is_set():
            path = _find_joycon(side)
            if path is None:
                logger.debug("Joy-Con %s not found, retrying in %.0fs…", side, RECONNECT_INTERVAL)
                self._stop.wait(RECONNECT_INTERVAL)
                continue

            try:
                dev = InputDevice(path)
                logger.info("Joy-Con %s connected: %s (%s)", side, dev.name, path)
            except Exception as exc:
                logger.warning("Failed to open Joy-Con %s at %s: %s", side, path, exc)
                self._stop.wait(RECONNECT_INTERVAL)
                continue

            stick_norm = 0.0
            last_time = time.monotonic()

            try:
                while not self._stop.is_set():
                    r, _, _ = select.select([dev], [], [], 0.05)
                    now = time.monotonic()
                    dt = now - last_time
                    last_time = now

                    for rd in r:
                        for event in rd.read():
                            if event.type == ecodes.EV_ABS and event.code == axis_code:
                                raw = max(min(event.value / 32767.0, 1.0), -1.0)
                                stick_norm = 0.0 if abs(raw) < self._deadzone else raw
                            elif event.type == ecodes.EV_KEY and event.code == toggle_code and event.value == 1:
                                self._toggle(side)

                    if abs(stick_norm) > 0:
                        delta = stick_norm * self._stick_speed * dt
                        self._update_velocity(side, delta)
            except Exception as exc:
                logger.warning("Joy-Con %s disconnected: %s", side, exc)

    def _toggle(self, side: str) -> None:
        mid = (self._open + self._close) / 2
        with self._lock:
            if side == "left":
                self._left_gripper = self._close if self._left_gripper < mid else self._open
            else:
                self._right_gripper = self._close if self._right_gripper < mid else self._open

    def _update_velocity(self, side: str, delta: float) -> None:
        with self._lock:
            if side == "left":
                self._left_gripper = max(self._lo, min(self._hi, self._left_gripper + delta))
            else:
                self._right_gripper = max(self._lo, min(self._hi, self._right_gripper + delta))
