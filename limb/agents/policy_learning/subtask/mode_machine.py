"""Sub-task mode machine — extends the DAgger phase pattern to sub-task RL.

Modes: VLA | RL | RESET | PAUSED | HUMAN | TERMINAL.

Two transition profiles (see ``limb_subtask_rl_integration.md`` §3):
- TRAINING (one sub-task, fast data): RESET<->RL<->HUMAN; RL->RESET on success.
- INFERENCE (full task, bidirectional): VLA<->RL; RL->VLA on success.

Transitions come from three sources: the coding-agent SELECTOR, the VERIFIER
predicates (success / unrecoverable / start-ok), and HUMAN pedal events. Programmatic
transitions are guarded against oscillation (min-mode-steps, cooldown). Thread-safe,
like ``DAggerEvents`` (human-event inbox consumed once per tick).
"""

from __future__ import annotations

import enum
from threading import Event, Lock


class SubtaskMode(enum.Enum):
    VLA = "vla"
    RL = "rl"
    RESET = "reset"
    PAUSED = "paused"
    HUMAN = "human"
    TERMINAL = "terminal"


# Allowed programmatic edges per profile (human pedal can always PAUSE/resume).
_TRANSITIONS: dict[str, set[tuple[SubtaskMode, SubtaskMode]]] = {
    "training": {
        (SubtaskMode.RESET, SubtaskMode.RL),
        # VLA APPROACH phase (user spec): RESET -> VLA (approach the vial) -> RL when
        # the coding-agent selector sees the gripper ABOVE the vial (bottleneck begins).
        (SubtaskMode.RESET, SubtaskMode.VLA),
        (SubtaskMode.VLA, SubtaskMode.RL),
        (SubtaskMode.VLA, SubtaskMode.HUMAN),
        (SubtaskMode.RL, SubtaskMode.RESET),
        (SubtaskMode.RL, SubtaskMode.HUMAN),
        # unrecoverable can fire DURING reset too (e.g. vial dropped mid-carry) — the
        # escalation edge must exist from RESET or the loop wedges there (review H7).
        (SubtaskMode.RESET, SubtaskMode.HUMAN),
        (SubtaskMode.HUMAN, SubtaskMode.RESET),
        # VLA-APPROACH LOOP (insert sub-task, user 2026-07-31): with the agent's
        # human_resume_to="vla", HUMAN hands to the VLA (pi0.5 re-grasps the staged
        # vial and carries it to the stand) and the selector fires VLA -> RL above
        # the stand EVERY episode — entry states match eval's VLA-approach
        # distribution by construction.
        (SubtaskMode.HUMAN, SubtaskMode.VLA),
        # HUMAN resumes DIRECTLY to RL (user rule 2026-07-07): the human guarantees
        # the sub-task start state, so no re-staging reset in between.
        (SubtaskMode.HUMAN, SubtaskMode.RL),
    },
    "inference": {
        (SubtaskMode.VLA, SubtaskMode.RL),
        (SubtaskMode.RL, SubtaskMode.VLA),
        (SubtaskMode.RL, SubtaskMode.HUMAN),
        (SubtaskMode.VLA, SubtaskMode.HUMAN),
        (SubtaskMode.HUMAN, SubtaskMode.VLA),
        (SubtaskMode.VLA, SubtaskMode.TERMINAL),
    },
}


class SubtaskModeMachine:
    def __init__(self, profile: str = "training", initial: SubtaskMode | None = None,
                 min_mode_steps: int = 5, cooldown_steps: int = 5):
        if profile not in _TRANSITIONS:
            raise ValueError(f"profile must be one of {list(_TRANSITIONS)}")
        self.profile = profile
        self._edges = _TRANSITIONS[profile]
        self._lock = Lock()
        self._mode = initial or (SubtaskMode.RESET if profile == "training" else SubtaskMode.VLA)
        self._steps_in_mode = 0
        self._cooldown = 0
        self.min_mode_steps = min_mode_steps
        self.cooldown_steps = cooldown_steps
        self._pending_human: str | None = None     # "pause_resume" | "to_human"
        self._prev_before_pause: SubtaskMode | None = None
        self.stop = Event()

    @property
    def mode(self) -> SubtaskMode:
        with self._lock:
            return self._mode

    @property
    def steps_in_mode(self) -> int:
        with self._lock:
            return self._steps_in_mode

    def tick(self) -> None:
        """Advance per-tick counters (call once per control step)."""
        with self._lock:
            self._steps_in_mode += 1
            if self._cooldown > 0:
                self._cooldown -= 1

    def request_human(self, event: str) -> None:
        """Human pedal: ``pause_resume`` (PAUSED<->prev) or ``to_human`` (-> HUMAN)."""
        with self._lock:
            self._pending_human = event

    def _force(self, new: SubtaskMode) -> tuple[SubtaskMode, SubtaskMode]:
        old, self._mode = self._mode, new
        self._steps_in_mode = 0
        self._cooldown = self.cooldown_steps
        return old, new

    def consume_human(self) -> tuple[SubtaskMode, SubtaskMode] | None:
        with self._lock:
            ev, self._pending_human = self._pending_human, None
            if ev is None:
                return None
            if ev == "to_human":
                return self._force(SubtaskMode.HUMAN)
            if ev == "pause_resume":
                if self._mode == SubtaskMode.PAUSED:
                    if self.profile == "inference":
                        # EVAL (user 2026-07-14): the operator pauses to restage — the
                        # arms are back at the INITIAL state on resume, so a paused
                        # mid-episode RL context is meaningless. Always resume into
                        # VLA (the full-task start); the selector re-fires RL when
                        # the gripper is above the next vial.
                        self._prev_before_pause = None
                        return self._force(SubtaskMode.VLA)
                    # Resume to the mode we paused from (review H8: previously nothing ever
                    # resumed — PAUSED was a one-way trap). Fall back to the profile start.
                    target = self._prev_before_pause or (
                        SubtaskMode.RESET if self.profile == "training" else SubtaskMode.VLA)
                    self._prev_before_pause = None
                    return self._force(target)
                self._prev_before_pause = self._mode
                return self._force(SubtaskMode.PAUSED)
            return None

    def propose(self, target: SubtaskMode, *, force: bool = False) -> tuple[SubtaskMode, SubtaskMode] | None:
        """Request a programmatic transition (from selector/verifier). Honoured only
        if the edge is allowed and guardrails pass (unless ``force``, e.g. verifier
        success/unrecoverable, which bypass min-steps/cooldown)."""
        with self._lock:
            if target == self._mode:
                return None
            edge = (self._mode, target)
            allowed = edge in self._edges or (self._mode == SubtaskMode.PAUSED)
            if not allowed:
                return None
            if not force:
                if self._steps_in_mode < self.min_mode_steps or self._cooldown > 0:
                    return None
            return self._force(target)

    def reset(self, mode: SubtaskMode | None = None) -> None:
        with self._lock:
            self._mode = mode or (SubtaskMode.RESET if self.profile == "training" else SubtaskMode.VLA)
            self._steps_in_mode = 0
            self._cooldown = 0
            self._pending_human = None
        self.stop.clear()
