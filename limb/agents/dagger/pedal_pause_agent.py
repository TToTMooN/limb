"""Pedal-gated two-phase policy wrapper: PAUSED <-> AUTONOMOUS (VLA).

DAggerAgent's pause function without the DAgger parts — no leader arms, no
bilateral teleop, no CORRECTING phase. For policy evaluation recording: the
operator stages the scene in PAUSED (robots hold), the pause/resume pedal
releases the policy, and the same pedal freezes it again for labeling and
re-staging. Boots PAUSED so the policy never drives an unstaged scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from loguru import logger

from limb.agents.agent import Agent
from limb.agents.constants import ActionSpec
from limb.agents.dagger.phase import DAggerEvents, DAggerPhase
from limb.agents.dagger.phase_trigger import PhaseTrigger
from limb.utils.portal_utils import remote


@dataclass
class PedalPausePolicyAgent(Agent):
    """Two-phase composite: ``inner_policy`` drives in AUTONOMOUS; PAUSED
    returns an action without robot entries so ``env.step`` commands nothing
    and every arm holds its last commanded position via PD (the same hold
    DAggerAgent uses).

    Each PAUSED -> AUTONOMOUS edge calls ``inner_policy.reset()`` so the
    first resumed chunk is inferred from the freshly staged scene, never a
    stale buffer (sync inference — same contract as DAggerAgent).

    The correction key of a double pedal is bounced back to PAUSED: there
    are no leaders in this stack, so CORRECTING is not reachable.
    """

    inner_policy: Agent = None  # set via _target_
    phase_trigger: PhaseTrigger = None  # set via _target_
    initial_phase: str = "paused"

    def __post_init__(self) -> None:
        if self.inner_policy is None or self.phase_trigger is None:
            raise ValueError("PedalPausePolicyAgent requires inner_policy and phase_trigger")
        boot_phase = DAggerPhase(self.initial_phase)
        if boot_phase is DAggerPhase.CORRECTING:
            raise ValueError("initial_phase must be 'paused' or 'autonomous'")
        self._events = DAggerEvents(initial_phase=boot_phase)
        self.phase_trigger.start(self._events)
        logger.info("PedalPausePolicyAgent started (initial phase: {})", boot_phase.value)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        transition = self._events.consume_transition()
        if transition is not None:
            _, new_phase = transition
            if new_phase is DAggerPhase.CORRECTING:
                self._events.reset(DAggerPhase.PAUSED)
                logger.info("Correction pedal ignored (no leaders in this stack) — staying PAUSED")
            elif new_phase is DAggerPhase.AUTONOMOUS and hasattr(self.inner_policy, "reset"):
                self.inner_policy.reset()

        phase = self._events.phase
        if phase is DAggerPhase.AUTONOMOUS:
            action: Dict[str, Any] = dict(self.inner_policy.act(obs))
        else:
            # No robot entries -> env._apply_action commands nothing; PD holds.
            action = {}
        # Recorder metadata (phase.npy); env._apply_action skips non-dict values.
        action["_phase"] = phase.value
        return action

    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        return self.inner_policy.action_spec()

    def reset(self) -> None:
        """Reset the phase machine to the boot phase and the inner policy."""
        self._events.reset(DAggerPhase(self.initial_phase))
        if hasattr(self.inner_policy, "reset"):
            self.inner_policy.reset()

    def close(self) -> None:
        for label, obj in (("phase_trigger", self.phase_trigger), ("inner_policy", self.inner_policy)):
            if obj is None or not hasattr(obj, "close"):
                continue
            try:
                obj.close()
            except Exception as e:
                logger.warning("PedalPausePolicyAgent: error closing {}: {}", label, e)
        logger.info("PedalPausePolicyAgent closed")

    @remote()
    def phase_name(self) -> str:
        """'paused' | 'autonomous' — polled by the launch loop for edge logs/TUI."""
        return self._events.phase.value

    @remote()
    def phase_log_label(self) -> str:
        return "Eval phase"
