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
    * ``gravity_comp_per_joint_offset`` / ``gravity_comp_per_joint_slope``
      — optional per-joint affine correction on top of the scaled gravity
      model, applied as ``g_i = scale_i * g_i + offset_i + slope_i * q_i``.
      Used when the URDF gravity *shape* (not just amplitude) doesn't
      match reality on a particular joint (e.g. a wrist roll whose link
      COM is offset from the URDF baseline).  See
      ``scripts/diagnostics/zero_g_record_j4.py`` for how these values
      are calibrated.  Default is no-op (all zeros).
    * ``gravity_comp_per_joint_damping`` — optional per-joint viscous
      damping ``-kd_i * qdot_i`` injected into the gravity-comp torque.
      Unlike ``damped_compliant_mode`` (which scales the motor's PD kd
      uniformly), this is a per-joint knob applied through the comp
      path, so it composes with ``zero_torque_mode`` and can target one
      joint without touching the rest.  Used to absorb residual
      gravity-model error on a specific joint (e.g. J4 on the leader).

      NOTE: i2rt's ``update()`` multiplies the full returned ``g`` by the
      global ``gravity_comp_factor`` before sending it to the motor (see
      ``dependencies/i2rt/i2rt/robots/motor_chain_robot.py:300``).  That
      means ``offset``, ``slope*q``, and ``-damping*qdot`` are all scaled
      by ``gravity_comp_factor`` too, not just the URDF-model term.  With
      ``gravity_comp_factor: 1.0`` (current leader configs) this is a
      no-op, but if you raise the factor, re-tune offset/slope/damping
      against the new effective scale.
    * ``position_mode`` — restores the PD gains saved at init.  This is
      the missing inverse of upstream's ``zero_torque_mode`` (which zeros
      ``_kp``/``_kd`` but doesn't expose a way back).  Required for
      DAgger leader arms that toggle between operator-backdriven
      (zero-torque) and follower-mirroring (position) modes.
    * ``gripper_spring_mode`` — PD spring on the gripper joint only,
      composes orthogonally with whatever mode the arm is in.  Used by
      DAgger to give the leader's gripper an auto-return-to-target feel
      during CORRECTING while the arm stays backdrivable.
    """

    def __init__(
        self,
        *args,
        gravity_comp_per_joint_scale: Optional[Sequence[float]] = None,
        gravity_comp_per_joint_offset: Optional[Sequence[float]] = None,
        gravity_comp_per_joint_slope: Optional[Sequence[float]] = None,
        gravity_comp_per_joint_damping: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> None:
        # IMPORTANT: ``super().__init__`` starts the i2rt motor-chain control
        # thread, which immediately begins calling our overridden
        # ``_compute_gravity_compensation``.  Set the sentinel attributes *before*
        # the super call or we race the thread on first tick.
        self._gc_per_joint: Optional[np.ndarray] = None
        self._gc_per_joint_offset: Optional[np.ndarray] = None
        self._gc_per_joint_slope: Optional[np.ndarray] = None
        self._gc_per_joint_damping: Optional[np.ndarray] = None

        super().__init__(*args, **kwargs)

        expected = len(self.motor_chain)

        def _check(name: str, vals: Sequence[float]) -> np.ndarray:
            arr = np.asarray(vals, dtype=np.float64)
            if arr.shape != (expected,):
                raise ValueError(
                    f"{name} must have length {expected} (motors in chain), "
                    f"got shape {arr.shape}"
                )
            return arr

        if gravity_comp_per_joint_scale is not None:
            self._gc_per_joint = _check("gravity_comp_per_joint_scale", gravity_comp_per_joint_scale)
            logging.info(f"{self}: gravity_comp_per_joint_scale = {self._gc_per_joint.tolist()}")
        if gravity_comp_per_joint_offset is not None:
            self._gc_per_joint_offset = _check("gravity_comp_per_joint_offset", gravity_comp_per_joint_offset)
            logging.info(f"{self}: gravity_comp_per_joint_offset = {self._gc_per_joint_offset.tolist()}")
        if gravity_comp_per_joint_slope is not None:
            self._gc_per_joint_slope = _check("gravity_comp_per_joint_slope", gravity_comp_per_joint_slope)
            logging.info(f"{self}: gravity_comp_per_joint_slope = {self._gc_per_joint_slope.tolist()}")
        if gravity_comp_per_joint_damping is not None:
            self._gc_per_joint_damping = _check("gravity_comp_per_joint_damping", gravity_comp_per_joint_damping)
            logging.info(f"{self}: gravity_comp_per_joint_damping = {self._gc_per_joint_damping.tolist()}")

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
        # Per-joint affine + damping corrections are all gated on joint_state
        # being available: the parent returns zeros when joint_state is None
        # (init race), and we don't want to apply a constant offset torque at
        # an unknown pose.
        if joint_state is not None:
            if self._gc_per_joint_offset is not None:
                g = g + self._gc_per_joint_offset
            if self._gc_per_joint_slope is not None:
                g = g + self._gc_per_joint_slope * joint_state.pos
            if self._gc_per_joint_damping is not None:
                g = g - self._gc_per_joint_damping * joint_state.vel
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
            # Clear any feedforward torque that an earlier mode might have
            # left in place (e.g. gripper_spring_mode's friction-comp bias).
            # Without this, the bias would keep pushing the gripper while the
            # arm tracks position commands.
            self._commands.torques[:] = 0.0
            self._kp = self._initial_kp.copy()
            self._kd = self._initial_kd.copy()

    def gripper_spring_mode(
        self,
        spring_target_rad: float,
        spring_kp: float,
        spring_kd: float,
        spring_torque_bias: float = 0.0,
    ) -> None:
        """Add a PD spring on the gripper joint only; arm joints untouched.

        Writes only the gripper slot of ``_kp`` / ``_kd`` / ``_commands.pos``
        / ``_commands.torques``.  Arm joint gains and commands are left
        exactly as the caller had them (typically ``zero_torque_mode`` or
        ``damped_compliant_mode`` set up immediately before), so the spring
        composes orthogonally with the arm's backdrive behavior.

        The motor's onboard PD law sustains the spring force autonomously
        once seeded — no per-tick re-command is required.  Used by
        :class:`DAggerAgent` during CORRECTING to give the leader's gripper
        an auto-return-to-target feel while keeping the arm free.

        Args:
            spring_target_rad: Gripper rest position in RAW motor radians
                (same units as ``gripper_limits`` in the arm's YAML, e.g.
                ``-4.55`` for a YAM trigger with ``gripper_limits:
                [0.0, -5.2]``).  Written directly to ``_commands.pos[gi]``;
                no remapping is performed inside this method.
            spring_kp:     PD stiffness on the gripper joint, in raw motor
                units (N·m per rad of motor displacement).
            spring_kd:     PD damping on the gripper joint, in raw motor
                units.
            spring_torque_bias: Constant feedforward Nm added to the gripper
                motor on top of the PD term.  Use to overcome static friction
                in the gripper linkage so the spring reliably returns to
                ``spring_target_rad`` instead of getting stuck near the
                operator's last hold point.  Sign convention matches the
                motor: with ``gripper_limits: [0.0, -5.2]`` (open is more
                negative), a *negative* bias pushes toward open.  Typical
                magnitudes are 0.02 to 0.1 Nm; too large and the gripper
                snaps open even against an operator squeeze.  Cleared on
                ``position_mode``/``zero_torque_mode`` so the bias only
                applies while the spring is active.
        """
        if self._gripper_index is None:
            raise RuntimeError(
                f"gripper_spring_mode requires a configured gripper_index on {self}"
            )
        gi = self._gripper_index
        target_raw = float(spring_target_rad)
        torque_bias = float(spring_torque_bias)
        logging.info(
            f"Entering gripper_spring_mode "
            f"(target_rad={target_raw:.3f}, kp={spring_kp}, kd={spring_kd}, "
            f"torque_bias={torque_bias:+.4f}) for {self}"
        )
        # The motor's update loop reads kp/kd from ``self._commands.kp/kd``
        # (deep-copied each tick), NOT from ``self._kp/_kd`` directly — the
        # latter is the "nominal stash" that ``command_joint_pos`` copies
        # into ``_commands`` on each command.  Because the leader isn't in
        # the action dict during CORRECTING (no ``command_joint_pos`` calls),
        # we must write the spring gains into BOTH places: ``_kp/_kd`` so a
        # later command_joint_pos propagates them, and ``_commands.kp/kd``
        # so they take effect *now*.
        with self._command_lock:
            self._kp[gi] = float(spring_kp)
            self._kd[gi] = float(spring_kd)
            self._commands.kp[gi] = float(spring_kp)
            self._commands.kd[gi] = float(spring_kd)
            self._commands.pos[gi] = target_raw
            # NOTE: This bias gets cleared on the next mode transition because
            # position_mode (above) explicitly zeros ``_commands.torques``, and
            # zero_torque_mode / damped_compliant_mode (inherited / overridden)
            # replace ``_commands`` with a freshly zeroed JointCommands. Keep
            # that invariant if you ever override those methods upstream or
            # add a new mode that writes ``_commands`` in place.
            self._commands.torques[gi] = torque_bias

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
