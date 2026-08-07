"""Rich TUI for limb control loop and data collection sessions.

Provides a clean status panel that coexists with loguru output.
When the TUI is active, log lines appear below the Live panel.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class SessionState:
    """Snapshot of data collection session state, pushed from session to TUI."""

    recording: bool = False
    episode_current: int = 0
    episode_total: int = 0
    episode_duration_s: float = 0.0
    countdown_remaining: float = 0.0
    episodes_successful: int = 0
    episodes_discarded: int = 0
    task_instruction: str = ""
    controls_hint: str = ""
    # DAgger-only fields (None / 0 for non-DAgger sessions)
    dagger_phase: Optional[str] = None  # "autonomous" / "paused" / "correcting"
    dagger_intervention: bool = False  # current tick's intervention flag
    dagger_intervention_count: int = 0  # interventions in current episode
    dagger_total_ticks: int = 0  # total ticks recorded in current episode


@dataclass
class StatusDisplay:
    """Rich Live panel showing control loop Hz and session state.

    Swaps the loguru sink so log lines render cleanly below the panel.
    """

    refresh_rate: float = 4.0

    _live: object = field(default=None, init=False, repr=False)
    _console: object = field(default=None, init=False, repr=False)
    _loguru_sink_id: Optional[int] = field(default=None, init=False, repr=False)
    _hz: float = field(default=0.0, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)
    _session: Optional[SessionState] = field(default=None, init=False, repr=False)
    _standalone_phase: Optional[str] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Start the Live panel and reroute loguru through Rich console."""
        from rich.console import Console
        from rich.live import Live

        self._console = Console(stderr=True)
        self._live = Live(
            self._build_panel(),
            console=self._console,
            refresh_per_second=self.refresh_rate,
            transient=False,
        )
        self._live.start()

        # Swap loguru sink: remove default stderr, add one that prints through Rich
        logger.remove()
        console = self._console

        def _rich_sink(message: str) -> None:
            # message already has loguru formatting; strip trailing newline for rich
            text = str(message).rstrip("\n")
            console.print(text, highlight=False, markup=False)

        self._loguru_sink_id = logger.add(
            _rich_sink,
            format="<level>{time:HH:mm:ss} | {level:<7} | {message}</level>",
            level="INFO",
            colorize=True,
            filter={"robocam.video_writer": "WARNING"},
        )

    def stop(self) -> None:
        """Stop the Live panel and restore normal loguru stderr sink."""
        if self._live is not None:
            self._live.stop()
            self._live = None

        # Restore normal loguru sink
        if self._loguru_sink_id is not None:
            logger.remove(self._loguru_sink_id)
            self._loguru_sink_id = None
        logger.add(
            sys.stderr,
            format="<level>{time:HH:mm:ss.SSS} | {level:<7} | {file}:{line} - {message}</level>",
            level="INFO",
            colorize=True,
            filter={"robocam.video_writer": "WARNING"},
        )

    def update_loop(self, hz: float, step: int) -> None:
        """Update Hz and step count, refresh the panel."""
        self._hz = hz
        self._step = step
        self._refresh()

    def update_phase(self, phase: Optional[str]) -> None:
        """Set the standalone DAgger phase string shown in the panel.

        Used when no DataCollectionSession is active (e.g. teleop without
        recording) so the operator can still see AUTONOMOUS / PAUSED /
        CORRECTING.  When a session *is* active, its SessionState push
        takes precedence.
        """
        self._standalone_phase = phase
        self._refresh()

    def update_session(self, state: SessionState) -> None:
        """Update session state and refresh the panel."""
        self._session = state
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_panel())

    def _build_panel(self) -> object:
        from rich.panel import Panel
        from rich.text import Text

        lines = Text()

        # Line 1: Hz | step | optional episode + recording status
        parts = [f"{self._hz:5.1f} Hz", f"step {self._step}"]

        s = self._session
        if s is not None:
            target = s.episode_total if s.episode_total > 0 else "\u221e"
            parts.append(f"Episode {s.episode_current}/{target}")

            if s.countdown_remaining > 0:
                parts.append(f"\u23f3 {s.countdown_remaining:.0f}s")
            elif s.recording:
                parts.append(f"\u25cf RECORDING {s.episode_duration_s:.0f}s")
            else:
                parts.append("IDLE")

        lines.append(" \u2502 ".join(parts))

        # Line 2: task instruction (if session active)
        if s is not None and s.task_instruction:
            lines.append("\n")
            task_display = s.task_instruction
            if len(task_display) > 70:
                task_display = task_display[:67] + "..."
            lines.append(f"Task: {task_display}")

        # Line 3: DAgger phase (and intervention rate when a session is active).
        # Session-pushed state takes precedence; otherwise fall back to the
        # standalone phase plumbed from launch.py.
        phase_str: Optional[str] = None
        if s is not None and s.dagger_phase is not None:
            phase_str = s.dagger_phase
        elif self._standalone_phase is not None:
            phase_str = self._standalone_phase

        if phase_str is not None:
            lines.append("\n")
            phase_label = phase_str.upper()
            color = {
                "AUTONOMOUS": "cyan",
                "PAUSED": "yellow",
                "CORRECTING": "magenta",
                # SubRL online-RL loop modes (SubtaskRLAgent.phase_name)
                "RL": "cyan",          # RL rollout: pi0.5 + residual driving
                "RESET": "green",      # coding-agent auto-reset (EAP place-back / staging)
                "HUMAN": "red",        # Call-Human: scene needs restoring
                "VLA": "cyan",
                "TERMINAL": "white",
            }.get(phase_label, "white")
            lines.append("phase: ", style="dim")
            lines.append(phase_label, style=f"bold {color}")
            if s is not None and s.dagger_total_ticks > 0:
                pct = 100.0 * s.dagger_intervention_count / s.dagger_total_ticks
                lines.append(
                    f"   interventions {s.dagger_intervention_count}/{s.dagger_total_ticks} ({pct:.1f}%)",
                    style="dim",
                )

        # Line 4: controls hint
        if s is not None and s.controls_hint:
            lines.append("\n")
            lines.append(s.controls_hint)

        return Panel(lines, title="limb", border_style="blue", expand=False)
