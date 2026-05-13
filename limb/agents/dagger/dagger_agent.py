"""DAggerAgent — composite Agent that switches between policy and teleop.

Wraps an ``inner_policy`` (e.g. :class:`YamPolicyAgent`) and an
``inner_teleop`` (e.g. :class:`YamYamBilateralAgent`) and dispatches
per-tick control to one of them based on a three-phase state machine
driven by a foot-pedal trigger.

For 4-YAM bimanual setups, this agent is also responsible for the
LEADER arms: it mirrors them onto the followers' commanded pose during
AUTONOMOUS, and toggles them between ``position`` and ``zero_torque``
modes on phase transitions so the operator can backdrive them in
CORRECTING.

See :mod:`limb.agents.dagger.phase` for the state machine and
:mod:`limb.agents.dagger.phase_trigger` for the input device protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
from dm_env.specs import Array
from loguru import logger

from limb.agents.agent import Agent
from limb.agents.dagger.phase import DAggerEvents, DAggerPhase
from limb.agents.dagger.phase_trigger import PhaseTrigger
from limb.utils.portal_utils import remote


@dataclass
class DAggerAgent(Agent):
    """Composite Agent wiring policy + bilateral teleop + phase trigger.

    Parameters
    ----------
    inner_policy : Agent
        Drives the followers in AUTONOMOUS phase.  Must implement
        ``act(obs) -> {follower_key: {"pos": (7,)}}`` and a public
        ``reset()`` that flushes any internal action buffer.
    inner_teleop : Agent
        Drives the followers in CORRECTING phase by reading leader joint
        state from ``obs``.  Typically
        :class:`YamYamBilateralAgent`.
    phase_trigger : PhaseTrigger
        Pushes ``pause_resume`` / ``correction`` events into the agent's
        :class:`DAggerEvents` inbox.  ``start()`` is called from
        ``__post_init__``; ``close()`` from :meth:`close`.
    follower_keys, leader_keys : Sequence[str]
        Robot keys for the two roles, in the same order so a leader
        mirrors its paired follower.
    """

    inner_policy: Agent = None  # set via _target_
    inner_teleop: Agent = None  # set via _target_
    phase_trigger: PhaseTrigger = None  # set via _target_
    follower_keys: Sequence[str] = field(default_factory=lambda: ("left", "right"))
    leader_keys: Sequence[str] = field(default_factory=lambda: ("leader_left", "leader_right"))

    # Linear blend ramp on PAUSED -> AUTONOMOUS.  At the resume tick the agent
    # captures the follower's current observed pose (where the operator left
    # it) and blends commanded actions from that pose to the policy's target
    # over ``resume_ramp_s`` seconds.  Set to 0.0 to disable (hard switch).
    # Leader mirror commands use the blended value so leader/follower stay
    # in lockstep during the ramp.
    resume_ramp_s: float = 0.5

    # Damping for the leader during CORRECTING.  When 0 (default), CORRECTING
    # entry sets the leaders to bare ``zero_torque_mode`` (kp=kd=0; gravity
    # comp still active).  When > 0, the agent uses
    # ``damped_compliant_mode(kd_scale)`` instead, which keeps kp=0 but
    # restores a fraction of the original kd to suppress ringing/overshoot.
    # Typical values: 0.05 - 0.20.  See YamMotorChainRobot.damped_compliant_mode.
    correcting_kd_scale: float = 0.0

    # On CORRECTING -> PAUSED, optionally re-stiffen the leader at its current
    # observed pose so it holds in place instead of staying limp / drifting
    # under residual gravity-comp imbalance.  Makes PAUSED a uniform
    # "everything frozen" state regardless of which phase you came from.
    pause_stiffens_leader: bool = True

    # Optional PD spring on the LEADER gripper joint during CORRECTING.
    # When ``gripper_spring_target_rad`` is set, the CORRECTING transition
    # layers a gripper-only PD spring on top of the arm mode (zero_torque or
    # compliant — driven by ``correcting_kd_scale``).  Arm joint behavior is
    # untouched; only the gripper slot of kp/kd/cmd is rewritten.  The
    # operator feels the gripper auto-return when they release the trigger,
    # while the arm remains fully backdrivable.  PAUSED-from-CORRECTING
    # clears the spring automatically via ``position_mode`` (which restores
    # the initial per-joint kp/kd).
    #
    # ``gripper_spring_target_rad`` is in RAW motor radians (same units as
    # ``gripper_limits`` in the leader's robot YAML).  Example: with the
    # YAM trigger calibrated to ``gripper_limits: [0.0, -5.2]``, a value of
    # ``-2.8`` is the spring rest, roughly 54% of the way to fully open.
    # ``gripper_spring_lower_rad`` is the closed-side bound of the leader's
    # effective operating range — operator squeezes can bring the leader
    # past this value but the rescale treats anything more positive than
    # ``lower_rad`` as "follower fully closed".
    # ``gripper_spring_open_rad`` is the leader's fully-open raw position
    # (the second element of its ``gripper_limits``).  Together with
    # ``lower_rad`` and ``target_rad`` it parameterizes the leader-to-
    # follower rescale in :meth:`_rescale_leader_gripper`.
    # ``kp``/``kd`` are in raw motor units.  Default ``None`` on the target
    # keeps the feature opt-in.  The non-None defaults below mirror the
    # tuned values in configs/yam_dagger_spring_test.yaml.
    gripper_spring_target_rad: Optional[float] = None
    gripper_spring_lower_rad: float = -0.42
    gripper_spring_open_rad: float = -5.2
    gripper_spring_kp: float = 0.15
    gripper_spring_kd: float = 0.03
    # Constant feedforward Nm added to the gripper motor while the spring is
    # active.  Compensates static friction in the gripper linkage so the
    # spring reliably returns to ``gripper_spring_target_rad`` instead of
    # stalling near the operator's last hold position.  Sign matches the
    # motor: with ``gripper_limits: [0.0, -5.2]`` (open is more negative), a
    # *negative* bias pushes toward open.  Cleared automatically on the
    # CORRECTING -> PAUSED transition.
    gripper_spring_torque_bias: float = -0.05

    # Phase the state machine boots into.  Default ``"autonomous"`` matches
    # the normal DAgger workflow (policy drives followers from tick 1).
    # Set to ``"paused"`` for tests that want to skip the policy roundtrip
    # entirely — from PAUSED the operator can only enter CORRECTING via
    # the correction pedal, so ``inner_policy`` is never invoked.  Useful
    # for bringing up the gripper-spring path without a policy server.
    initial_phase: str = "autonomous"

    use_joint_state_as_action: bool = False

    def __post_init__(self) -> None:
        self.follower_keys = tuple(self.follower_keys)
        self.leader_keys = tuple(self.leader_keys)
        if len(self.follower_keys) != len(self.leader_keys):
            raise ValueError(
                f"follower_keys ({self.follower_keys}) and leader_keys ({self.leader_keys}) must have the same length"
            )
        if self.inner_policy is None or self.inner_teleop is None or self.phase_trigger is None:
            raise ValueError("DAggerAgent requires inner_policy, inner_teleop, and phase_trigger")

        try:
            boot_phase = DAggerPhase(self.initial_phase)
        except ValueError as e:
            raise ValueError(
                f"initial_phase must be one of {[p.value for p in DAggerPhase]}; "
                f"got {self.initial_phase!r}"
            ) from e
        self._events = DAggerEvents(initial_phase=boot_phase)
        self.phase_trigger.start(self._events)

        # Leaders are released by `release_at_startup` in launch_utils.  On the
        # first AUTONOMOUS tick we promote them back to position mode so they
        # can stiff-track the followers.  Subsequent mode payloads are queued
        # by phase transitions in `_apply_transition`.  Stored as a dict so
        # parameterized modes (e.g. compliant + kd_scale) can carry their args.
        self._leader_mode_payload: Optional[Dict[str, Any]] = {"mode": "position"}

        # Per-correction stats — logged on CORRECTING->PAUSED edge.
        self._correction_started_at: Optional[float] = None
        self._correction_tick_count: int = 0
        self._correction_index: int = 0

        # Resume-ramp state — populated on PAUSED -> AUTONOMOUS, cleared once the
        # blend has elapsed.  ``None`` means no ramp is currently active.
        self._ramp_start_time: Optional[float] = None
        self._ramp_start_pose: Optional[Dict[str, np.ndarray]] = None

        # Precompute gripper-spring rescale constants.  Validating here (rather
        # than per-tick in _rescale_leader_gripper) surfaces misconfiguration
        # at launch time instead of silently disabling the rescale at 100 Hz.
        self._spring_enabled: bool = False
        self._spring_lower_norm: float = 0.0
        self._spring_span: float = 1.0
        if self.gripper_spring_target_rad is not None:
            open_rad = float(self.gripper_spring_open_rad)
            if open_rad == 0.0:
                raise ValueError("gripper_spring_open_rad must be non-zero")
            target_norm = float(self.gripper_spring_target_rad) / open_rad
            lower_norm = float(self.gripper_spring_lower_rad) / open_rad
            if not (0.0 <= lower_norm < target_norm <= 1.0):
                raise ValueError(
                    "gripper spring bounds invalid: normalized "
                    f"lower={lower_norm:.4f}, target={target_norm:.4f} "
                    "must satisfy 0 <= lower < target <= 1 "
                    f"(rads: lower={self.gripper_spring_lower_rad}, "
                    f"target={self.gripper_spring_target_rad}, "
                    f"open={self.gripper_spring_open_rad})"
                )
            self._spring_enabled = True
            self._spring_lower_norm = lower_norm
            self._spring_span = target_norm - lower_norm

        logger.info(
            "DAggerAgent: followers={} leaders={} initial_phase={}",
            list(self.follower_keys),
            list(self.leader_keys),
            self._events.phase.value,
        )

    # -------------------------------------------------------------- #
    #  Agent protocol                                                #
    # -------------------------------------------------------------- #

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        # 1) Consume any pending phase transition and run its side effects.
        transition = self._events.consume_transition()
        if transition is not None:
            old_phase, new_phase = transition
            self._apply_transition(old_phase, new_phase, obs)

        # 2) Dispatch to the appropriate inner agent based on current phase.
        phase = self._events.phase
        if phase is DAggerPhase.AUTONOMOUS:
            action = self._dispatch_autonomous(obs)
        elif phase is DAggerPhase.PAUSED:
            # Empty dict — env.step skips _apply_action when len(action) == 0
            # so all robots hold their last commanded position via PD.
            action = {}
        else:  # CORRECTING
            action = self._dispatch_correcting(obs)
            self._correction_tick_count += 1

        # 3) Apply any pending leader mode-switch (init or transition).
        if self._leader_mode_payload is not None:
            for lkey in self.leader_keys:
                entry = action.setdefault(lkey, {})
                entry.update(self._leader_mode_payload)
            self._leader_mode_payload = None

        # 4) Stamp phase metadata for the recorder. Underscore-prefixed keys
        # are ignored by env._apply_action (it only consumes per-arm dicts).
        # The recorder strips them out of the action it stores and writes them
        # to phase.npy / correction_index.npy.
        action["_phase"] = phase.value           # "autonomous" | "paused" | "correcting"
        action["_correction_index"] = self._correction_index

        return action

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        # All 4 arms are commandable: followers get policy/teleop targets,
        # leaders get mirror targets in AUTONOMOUS.
        spec: Dict[str, Dict[str, Array]] = {}
        for key in tuple(self.follower_keys) + tuple(self.leader_keys):
            spec[key] = {"pos": Array(shape=(7,), dtype=np.float32)}
        return spec

    def reset(self) -> None:
        """Reset the phase machine and the inner policy.  Does NOT reset hardware."""
        self._events.reset(DAggerPhase.AUTONOMOUS)
        # Queue a leader -> position payload so the next act() restores PD
        # gains on the leaders.  Reset typically fires after a CORRECTING
        # handoff that put the leaders in zero-torque mode; without this the
        # mirrored `pos` commands wouldn't actually hold the leaders.
        self._leader_mode_payload = {"mode": "position"}
        if hasattr(self.inner_policy, "reset"):
            self.inner_policy.reset()

    def close(self) -> None:
        for label, obj in (
            ("phase_trigger", self.phase_trigger),
            ("inner_policy", self.inner_policy),
            ("inner_teleop", self.inner_teleop),
        ):
            if obj is None or not hasattr(obj, "close"):
                continue
            try:
                obj.close()
            except Exception as e:
                logger.warning("DAggerAgent: error closing {}: {}", label, e)
        logger.info("DAggerAgent closed")

    # -------------------------------------------------------------- #
    #  Read-only views (used by the recorder + UI)                   #
    # -------------------------------------------------------------- #
    #
    # These are exposed as @remote() *methods* (not @property) so they
    # can be queried via Portal RPC.  Properties don't survive the
    # Client/Server boundary — see limb.utils.portal_utils.Client.

    @remote()
    def is_correcting(self) -> bool:
        """True iff the current phase is CORRECTING (operator drives followers)."""
        return self._events.phase is DAggerPhase.CORRECTING

    @remote()
    def phase_name(self) -> str:
        """Current phase as a string: 'autonomous' / 'paused' / 'correcting'."""
        return self._events.phase.value

    # -------------------------------------------------------------- #
    #  Internal                                                      #
    # -------------------------------------------------------------- #

    def _apply_transition(
        self,
        old_phase: DAggerPhase,
        new_phase: DAggerPhase,
        obs: Dict[str, Any],
    ) -> None:
        logger.info("DAgger phase: {} -> {}", old_phase.value, new_phase.value)

        # Per-correction stats: track entry/exit of CORRECTING.
        if new_phase is DAggerPhase.CORRECTING:
            self._correction_started_at = time.monotonic()
            self._correction_tick_count = 0
            self._correction_index += 1
            logger.info("Correction #{} started", self._correction_index)
        elif old_phase is DAggerPhase.CORRECTING and self._correction_started_at is not None:
            duration = time.monotonic() - self._correction_started_at
            logger.info(
                "Correction #{} ended (duration={:.1f}s, ticks={})",
                self._correction_index,
                duration,
                self._correction_tick_count,
            )
            self._correction_started_at = None

        if new_phase is DAggerPhase.CORRECTING:
            # Hand the leaders to the operator.  Followers continue to track
            # whatever inner_teleop returns this tick.  Use damped-compliant
            # if the operator opted into it; otherwise bare zero-torque.
            if self.correcting_kd_scale > 0.0:
                payload: Dict[str, Any] = {
                    "mode": "compliant",
                    "kd_scale": float(self.correcting_kd_scale),
                }
            else:
                payload = {"mode": "zero_torque"}
            # Optional gripper PD spring layered on top of the arm mode.
            # Gripper-only — does not affect arm kp/kd/commands.
            if self.gripper_spring_target_rad is not None:
                payload["gripper_spring"] = {
                    "target_rad": float(self.gripper_spring_target_rad),
                    "kp": float(self.gripper_spring_kp),
                    "kd": float(self.gripper_spring_kd),
                    "torque_bias": float(self.gripper_spring_torque_bias),
                }
            self._leader_mode_payload = payload

        elif old_phase is DAggerPhase.CORRECTING and new_phase is DAggerPhase.PAUSED and self.pause_stiffens_leader:
            # Re-stiffen the leader at its current observed pose so it doesn't
            # drift while the operator decides what to do next.  position_mode
            # seeds the command target with the current pose, so there is no
            # snap toward a stale target — at most a small "click" if the
            # operator is still gripping the leader at the moment of the
            # transition.
            self._leader_mode_payload = {"mode": "position"}

        elif new_phase is DAggerPhase.AUTONOMOUS:
            # Coming back from PAUSED.  Stale chunks from before the takeover
            # would replay if we didn't flush them, so reset the policy first.
            if hasattr(self.inner_policy, "reset"):
                try:
                    self.inner_policy.reset()
                except Exception as e:
                    logger.warning("inner_policy.reset() failed: {}", e)
            # Promote leaders back to position mode so they can stiff-track
            # followers from this tick onward.
            self._leader_mode_payload = {"mode": "position"}

            # Capture the follower's current pose so _dispatch_autonomous
            # can blend from it to the policy's target.  Skipped when the
            # ramp is disabled (resume_ramp_s == 0) or the obs is missing
            # any follower's joint state.
            if self.resume_ramp_s > 0.0:
                start_pose: Dict[str, np.ndarray] = {}
                for fkey in self.follower_keys:
                    arm_obs = obs.get(fkey)
                    if not isinstance(arm_obs, dict):
                        continue
                    joint_pos = arm_obs.get("joint_pos")
                    gripper_pos = arm_obs.get("gripper_pos")
                    if joint_pos is None:
                        continue
                    jp = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
                    if gripper_pos is not None:
                        gp = np.asarray(gripper_pos, dtype=np.float64).reshape(-1)
                        start_pose[fkey] = np.concatenate([jp, gp])
                    else:
                        start_pose[fkey] = jp
                if start_pose:
                    self._ramp_start_time = time.monotonic()
                    self._ramp_start_pose = start_pose
                    logger.info(
                        "Resume blend ramp armed: {:.2f}s, {} arm(s)",
                        self.resume_ramp_s,
                        len(start_pose),
                    )

        # PAUSED: no leader mode change — they stay in whatever mode they
        # were in.  AUTO->PAUSED keeps leaders position-tracking (followers
        # also hold last pos, so leaders just hold their last mirror).
        # CORRECTING->PAUSED leaves leaders released; the operator either
        # holds them or they sag onto gravity-comp equilibrium.

    def _dispatch_autonomous(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """AUTONOMOUS: policy drives followers; mirror followers' targets onto leaders.

        For RECAP-style training we also stash the *unblended* policy target on
        each follower entry as ``policy_pos``.  During the resume blend window
        ``pos`` (commanded) differs from ``policy_pos`` (raw policy output);
        outside the blend they are equal.  The recorder picks ``policy_pos``
        up so the dataset preserves both streams.
        """
        follower_action = self.inner_policy.act(obs)
        action: Dict[str, Dict[str, Any]] = dict(follower_action) if follower_action else {}

        # Snapshot the policy's raw target before we touch it. Used as
        # `policy_pos` below so the recorder can distinguish blended vs
        # un-blended commands.
        policy_targets: Dict[str, np.ndarray] = {}
        for fkey in self.follower_keys:
            fentry = action.get(fkey)
            if isinstance(fentry, dict) and "pos" in fentry:
                policy_targets[fkey] = np.asarray(fentry["pos"]).copy()

        # Apply the post-resume blend ramp.  Each armed follower's command is
        # weighted: alpha * operator_end_pose + (1 - alpha) * policy_target,
        # with alpha decaying from 1.0 to 0.0 over resume_ramp_s.  Once the
        # ramp expires we drop the captured state and stop blending.
        if self._ramp_start_pose is not None and self._ramp_start_time is not None:
            elapsed = time.monotonic() - self._ramp_start_time
            if elapsed >= self.resume_ramp_s:
                logger.info("Resume blend ramp complete ({:.2f}s)", elapsed)
                self._ramp_start_pose = None
                self._ramp_start_time = None
            else:
                alpha = 1.0 - (elapsed / self.resume_ramp_s)
                for fkey, start_pos in self._ramp_start_pose.items():
                    fentry = action.get(fkey)
                    if not isinstance(fentry, dict) or "pos" not in fentry:
                        continue
                    target = np.asarray(fentry["pos"], dtype=np.float64).reshape(-1)
                    if target.shape != start_pos.shape:
                        # Dimension mismatch — skip blend for this arm rather
                        # than emit a malformed action.
                        continue
                    blended = alpha * start_pos + (1.0 - alpha) * target
                    action[fkey] = {**fentry, "pos": blended}

        # Re-stamp the policy_pos field after blending so each follower entry
        # exposes both streams. Done last so it survives the blend overwrite.
        for fkey, policy_target in policy_targets.items():
            fentry = action.get(fkey)
            if isinstance(fentry, dict):
                fentry["policy_pos"] = policy_target

        # Mirror leaders to whatever the (possibly blended) follower target is.
        for fkey, lkey in zip(self.follower_keys, self.leader_keys, strict=True):
            fentry = action.get(fkey)
            if fentry is None or "pos" not in fentry:
                continue
            action[lkey] = {"pos": fentry["pos"]}
        return action

    def _dispatch_correcting(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """CORRECTING: bilateral teleop translates leader joint state to follower commands.

        Leaders are released (zero_torque) and not in the action dict, so the
        env's PD controllers won't fight the operator.

        When the gripper spring is active, the leader's effective operating
        range shrinks to ``[gripper_spring_lower, gripper_spring_target]``
        (both in normalized [0, 1] obs space): the spring pulls the trigger
        back to ``gripper_spring_target`` on release, and ``lower`` is the
        closed-side bound of the operator's effective squeeze (squeezes
        past it clip).  Without correction the follower would only open to
        ``gripper_spring_target`` of its mechanical range.  We pass a
        shallow copy of ``obs`` with the leader's ``gripper_pos`` rescaled
        from ``[lower, target]`` onto ``[0, 1]`` (clipped) into the
        bilateral agent so the follower sees a full sweep across the
        operator's full trigger throw.  The original ``obs`` is unchanged
        so the recorder still stores the raw leader gripper reading.
        """
        return self.inner_teleop.act(self._rescale_leader_gripper(obs))

    def _rescale_leader_gripper(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Map the leader's effective ``[lower_norm, target_norm]`` trigger
        throw onto the follower's full ``[0, 1]`` command space.

        Constants (`_spring_lower_norm`, `_spring_span`) and the
        enabled/disabled gate are computed once in ``__post_init__`` so this
        runs flat per tick.  When disabled, returns ``obs`` unchanged.
        """
        if not self._spring_enabled:
            return obs
        new_obs: Dict[str, Any] = dict(obs)
        for lkey in self.leader_keys:
            arm = new_obs.get(lkey)
            if not isinstance(arm, dict):
                continue
            grip = arm.get("gripper_pos")
            if grip is None:
                continue
            rescaled = np.clip(
                (np.asarray(grip, dtype=np.float64) - self._spring_lower_norm) / self._spring_span,
                0.0,
                1.0,
            )
            new_obs[lkey] = {**arm, "gripper_pos": rescaled}
        return new_obs
