"""DAgger phase state machine.

Three observable phases:

* ``AUTONOMOUS`` — policy drives followers; leaders mirror followers.
* ``PAUSED``     — engine paused; all robots hold position.
* ``CORRECTING`` — operator drives leaders; followers track leaders.

Valid transitions (events come from the phase trigger):

    AUTONOMOUS  --pause_resume-->  PAUSED       (and back)
    PAUSED      --correction--->   CORRECTING   (and back)

You cannot go AUTONOMOUS<->CORRECTING directly: a PAUSED step in between
guarantees the operator is positioned and ready before control changes
hands.
"""

from __future__ import annotations

import enum
from threading import Event, Lock


class DAggerPhase(enum.Enum):
    AUTONOMOUS = "autonomous"
    PAUSED = "paused"
    CORRECTING = "correcting"


# (current_phase, event_name) -> next_phase.  Anything missing is rejected.
_DAGGER_TRANSITIONS: dict[tuple[DAggerPhase, str], DAggerPhase] = {
    (DAggerPhase.AUTONOMOUS, "pause_resume"): DAggerPhase.PAUSED,
    (DAggerPhase.PAUSED, "pause_resume"): DAggerPhase.AUTONOMOUS,
    (DAggerPhase.PAUSED, "correction"): DAggerPhase.CORRECTING,
    (DAggerPhase.CORRECTING, "correction"): DAggerPhase.PAUSED,
}


class DAggerEvents:
    """Thread-safe inbox for phase-trigger events.

    Trigger threads (foot pedal listener, etc.) call
    :meth:`request_transition` to enqueue an event.  The control loop
    calls :meth:`consume_transition` once per tick to atomically
    apply a pending event and read out the resulting transition, if any.

    Invalid transitions (e.g. AUTONOMOUS + ``correction``) are silently
    dropped at request time so the trigger thread cannot drive the
    machine into an inconsistent state.
    """

    def __init__(self, initial_phase: DAggerPhase = DAggerPhase.AUTONOMOUS) -> None:
        self._lock = Lock()
        self._phase = initial_phase
        self._pending: str | None = None
        # Session-level flags (used by upstream recorders / launch loops).
        self.stop_recording = Event()

    @property
    def phase(self) -> DAggerPhase:
        with self._lock:
            return self._phase

    def request_transition(self, event: str) -> bool:
        """Enqueue a transition request.  Returns ``True`` if accepted.

        Only events that correspond to a *valid* edge from the current
        phase are accepted; others are dropped.  If a request is already
        pending, it is overwritten — the most recent button press wins.
        """
        with self._lock:
            if (self._phase, event) not in _DAGGER_TRANSITIONS:
                return False
            self._pending = event
            return True

    def consume_transition(self) -> tuple[DAggerPhase, DAggerPhase] | None:
        """Atomically apply any pending transition.

        Returns ``(old_phase, new_phase)`` if a transition fired this tick,
        or ``None`` otherwise.  Phase mutation and pending-clear happen
        under the same lock so trigger threads can't race.
        """
        with self._lock:
            if self._pending is None:
                return None
            key = (self._phase, self._pending)
            self._pending = None
            new_phase = _DAGGER_TRANSITIONS.get(key)
            if new_phase is None:
                return None
            old_phase = self._phase
            self._phase = new_phase
            return old_phase, new_phase

    def reset(self, phase: DAggerPhase = DAggerPhase.AUTONOMOUS) -> None:
        """Clear any pending event and force the phase back to ``phase``."""
        with self._lock:
            self._phase = phase
            self._pending = None
        self.stop_recording.clear()
