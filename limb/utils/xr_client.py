"""
Thin wrapper around xrobotoolkit_sdk for VR controller input.
Requires XRoboToolkit PC Service to be running.

Install the SDK with: bash scripts/install_xrobotoolkit_sdk.sh
"""

from typing import Optional

import numpy as np

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    raise ImportError(
        "xrobotoolkit_sdk not installed. Run: bash scripts/install_xrobotoolkit_sdk.sh"
    )


class XrClient:
    """Client for reading VR controller poses, buttons, and triggers from XRoboToolkit SDK."""

    def __init__(self) -> None:
        xrt.init()
        print("XRoboToolkit SDK initialized.")

    def get_pose(self, name: str) -> np.ndarray:
        """Get device pose as [x, y, z, qx, qy, qz, qw].

        Args:
            name: One of "left_controller", "right_controller", "headset".
        """
        if name == "left_controller":
            return np.array(xrt.get_left_controller_pose())
        elif name == "right_controller":
            return np.array(xrt.get_right_controller_pose())
        elif name == "headset":
            return np.array(xrt.get_headset_pose())
        raise ValueError(f"Invalid pose name: {name}")

    def get_trigger(self, side: str) -> float:
        """Get analog trigger value (0.0 to 1.0). Side: 'left' or 'right'."""
        if side == "left":
            return float(xrt.get_left_trigger())
        elif side == "right":
            return float(xrt.get_right_trigger())
        raise ValueError(f"Invalid side: {side}")

    def get_grip(self, side: str) -> float:
        """Get analog grip value (0.0 to 1.0). Side: 'left' or 'right'."""
        if side == "left":
            return float(xrt.get_left_grip())
        elif side == "right":
            return float(xrt.get_right_grip())
        raise ValueError(f"Invalid side: {side}")

    def get_button(self, name: str) -> bool:
        """Get button state. Names: 'A', 'B', 'X', 'Y'."""
        lookup = {
            "A": xrt.get_A_button,
            "B": xrt.get_B_button,
            "X": xrt.get_X_button,
            "Y": xrt.get_Y_button,
        }
        if name in lookup:
            return bool(lookup[name]())
        raise ValueError(f"Invalid button: {name}")

    def close(self) -> None:
        xrt.close()
