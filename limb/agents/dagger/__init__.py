"""DAgger: human-in-the-loop teleop/policy switching for limb.

The composite :class:`DAggerAgent` wraps an inner policy agent + an inner
teleop agent + a :class:`PhaseTrigger` and dispatches per-tick control to
one of them based on a three-state machine
(AUTONOMOUS / PAUSED / CORRECTING).

This package only contains the orchestration layer.  Recording-side
plumbing (intervention tagging, episode segmentation) lives in
:mod:`limb.recording`; the inner policy and teleop are ordinary
:class:`limb.agents.agent.Agent` implementations.
"""

from limb.agents.dagger.dagger_agent import DAggerAgent
from limb.agents.dagger.pedal_pause_agent import PedalPausePolicyAgent
from limb.agents.dagger.phase import DAggerEvents, DAggerPhase
from limb.agents.dagger.phase_trigger import (
    FootPedalPhaseTrigger,
    PhaseTrigger,
)

__all__ = [
    "DAggerAgent",
    "DAggerEvents",
    "DAggerPhase",
    "FootPedalPhaseTrigger",
    "PedalPausePolicyAgent",
    "PhaseTrigger",
]
