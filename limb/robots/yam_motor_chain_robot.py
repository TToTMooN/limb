"""
Thin wrapper around MotorChainRobot that adds soft_release for graceful shutdown.
"""

import logging
import time

from i2rt.robots.motor_chain_robot import MotorChainRobot


class YamMotorChainRobot(MotorChainRobot):
    """MotorChainRobot with a soft_release method for gradual power-down."""

    def soft_release(self, duration_s: float = 2.0, steps: int = 50) -> None:
        """Gradually reduce gravity compensation then enter zero-torque mode.

        First disables PD tracking (kp/kd -> 0) so the arm is only held by
        gravity comp, then linearly ramps gravity_comp_factor from its current
        value to 0 so the arm lowers softly under gravity instead of dropping.
        """
        logging.info(f"Soft release over {duration_s:.1f}s for {self}")
        self.zero_torque_mode()

        initial_factor = self.gravity_comp_factor
        for i in range(steps + 1):
            self.gravity_comp_factor = initial_factor * (1.0 - i / steps)
            time.sleep(duration_s / steps)

        self.gravity_comp_factor = 0.0
