"""
Thin wrapper around MotorChainRobot that adds soft_release for graceful
shutdown, per-joint scaling on gravity comp, and a ``position_mode`` to
restore PD gains after ``zero_torque_mode``.
"""

import logging
import time
from typing import Optional, Sequence

import numpy as np
from i2rt.robots.motor_chain_robot import JointCommands, MotorChainRobot


class YamMotorChainRobot(MotorChainRobot):
    """MotorChainRobot extensions for limb.

    Adds three features over the upstream class:

    * ``soft_release`` — gradual gravity-comp ramp-down for graceful
      shutdown.
    * ``gravity_comp_per_joint_scale`` — optional per-joint multiplier on
      the gravity torque vector, applied *before* the global
      ``gravity_comp_factor``.  Useful when one joint sags more than
      others and you want to compensate that joint specifically without
      raising current draw on the rest of the chain.
    * ``position_mode`` — restores the PD gains saved at init.  This is
      the missing inverse of upstream's ``zero_torque_mode`` (which zeros
      ``_kp``/``_kd`` but doesn't expose a way back).  Required for
      DAgger leader arms that toggle between operator-backdriven
      (zero-torque) and follower-mirroring (position) modes.
    """

    def __init__(
        self,
        *args,
        gravity_comp_per_joint_scale: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> None:
        # IMPORTANT: ``super().__init__`` starts the i2rt motor-chain control
        # thread, which immediately begins calling our overridden
        # ``_compute_gravity_compensation``.  Set the sentinel attribute *before*
        # the super call or we race the thread on first tick.
        self._gc_per_joint: Optional[np.ndarray] = None

        super().__init__(*args, **kwargs)

        if gravity_comp_per_joint_scale is not None:
            scale = np.asarray(gravity_comp_per_joint_scale, dtype=np.float64)
            expected = len(self.motor_chain)
            if scale.shape != (expected,):
                raise ValueError(
                    f"gravity_comp_per_joint_scale must have length {expected} "
                    f"(motors in chain), got shape {scale.shape}"
                )
            self._gc_per_joint = scale
            logging.info(f"{self}: gravity_comp_per_joint_scale = {scale.tolist()}")

        # Snapshot the PD gains AFTER super().__init__ has applied the
        # YAML-configured kp/kd, so position_mode() can restore them later.
        # The control thread doesn't read these attributes so the late
        # assignment is safe.
        self._initial_kp = np.asarray(self._kp, dtype=np.float64).copy()
        self._initial_kd = np.asarray(self._kd, dtype=np.float64).copy()

    def _compute_gravity_compensation(self, joint_state: object) -> np.ndarray:
        g = super()._compute_gravity_compensation(joint_state)
        if self._gc_per_joint is not None:
            g = g * self._gc_per_joint
        return g

    def damped_compliant_mode(self, kd_scale: float = 0.1) -> None:
        """Backdriveable mode with viscous damping.

        Sets ``kp=0`` (no position tracking) but ``kd = initial_kd * kd_scale``
        so motion is opposed by velocity-proportional torque.  Gravity comp
        remains active, so the arm floats; the operator can backdrive it
        freely but ringing/overshoot is damped.

        ``kd_scale`` is typically 0.05-0.20.  Tune by feel:
            * too high  -> arm feels syrupy, hard to make small fast moves
            * too low   -> ringing remains, overshoot on release

        Used by :class:`DAggerAgent` during CORRECTING when
        ``correcting_kd_scale > 0`` is set on the agent.  Pure
        ``zero_torque_mode`` remains the default for backwards compatibility.
        """
        if kd_scale < 0:
            raise ValueError(f"kd_scale must be non-negative, got {kd_scale}")
        logging.info(f"Entering damped_compliant_mode (kd_scale={kd_scale:.3f}) for {self}")
        with self._command_lock:
            # Same target reset as zero_torque_mode so any stale position
            # target doesn't get servoed by the new (small but non-zero) Kd.
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._kp = np.zeros(len(self.motor_chain))
            self._kd = self._initial_kd * float(kd_scale)

    def position_mode(self) -> None:
        """Restore the PD gains saved at init.  Inverse of ``zero_torque_mode``.

        Seeds the position command with the current observed joint position
        before restoring gains, so the PD doesn't lurch toward whatever stale
        target ``zero_torque_mode`` left in ``_commands.pos`` (typically zeros).
        After this call, the arm holds its current pose with full stiffness;
        subsequent ``command_joint_pos`` calls track from there.
        """
        logging.info(f"Entering position_mode for {self}")
        with self._state_lock:
            current_pos = self._joint_state.pos.copy()
        with self._command_lock:
            # Mutate in place so we don't disturb other JointCommands fields.
            self._commands.pos = current_pos
            self._kp = self._initial_kp.copy()
            self._kd = self._initial_kd.copy()

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
