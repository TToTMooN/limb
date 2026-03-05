"""
Standalone Joy-Con gripper diagnostic — no robot hardware needed.

Verifies that paired Nintendo Joy-Con (L + R) controllers are detected via
Bluetooth/evdev and that stick + button inputs correctly produce gripper
values.

Usage:
  # Default settings (range 0.0–2.4, stick speed 3.0/s):
  uv run scripts/test_joycon_gripper.py

  # Custom gripper range and speed:
  uv run scripts/test_joycon_gripper.py --close 1.0 --speed 2.0

  # Print every N seconds instead of 10 Hz:
  uv run scripts/test_joycon_gripper.py --hz 5

Prerequisites:
  - Joy-Con (L) and/or Joy-Con (R) paired via Bluetooth
  - ``evdev`` Python package installed
  - User must have read access to /dev/input/event* (udev rule or sudo)
"""

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Joy-Con gripper input")
    parser.add_argument("--open", type=float, default=0.0, help="Gripper open value (default 0.0)")
    parser.add_argument("--close", type=float, default=2.4, help="Gripper close value (default 2.4)")
    parser.add_argument("--speed", type=float, default=3.0, help="Stick velocity in units/s (default 3.0)")
    parser.add_argument("--deadzone", type=float, default=0.05, help="Stick deadzone (default 0.05)")
    parser.add_argument("--hz", type=float, default=10.0, help="Print rate in Hz (default 10)")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    from limb.devices.joycon_gripper_reader import JoyConGripperReader

    reader = JoyConGripperReader(
        gripper_open=args.open,
        gripper_close=args.close,
        stick_speed=args.speed,
        deadzone=args.deadzone,
    )

    dt = 1.0 / args.hz
    print(f"\nReading Joy-Con gripper values at {args.hz:.0f} Hz.  Ctrl+C to stop.\n")
    print(f"  Stick Y  → proportional velocity (speed={args.speed:.1f}/s)")
    print(f"  ZL / ZR  → toggle open ({args.open:.1f}) / close ({args.close:.1f})\n")

    try:
        while True:
            left, right = reader.get_gripper_values()
            print(f"\r  L: {left:6.3f}   R: {right:6.3f}", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
