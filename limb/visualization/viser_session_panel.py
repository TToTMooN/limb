"""Viser GUI panel for data collection session management.

Replaces the Rich terminal TUI (StatusDisplay) when a viser server is
available. Renders session status + interactive buttons in the viser
sidebar. Also implements the TriggerSource protocol so GUI buttons can
feed signals into the session's trigger pipeline.

The panel attaches to an existing ViserServer (from ViserMonitor or a
standalone server) — it never creates its own.

Layout in the viser sidebar::

    ┌─ Data Collection ──────────────────┐
    │  100.2 Hz  │  step 4521            │
    │  Episode 3/10  │  ● RECORDING 12s  │
    │  Task: pick up red cube            │
    │                                    │
    │  [▶ Start/Stop]  [✓ Success]       │
    │  [✗ Discard]     [⏹ Quit]         │
    │                                    │
    │  Controls: [SPACE] toggle ...      │
    └────────────────────────────────────┘
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Optional

import viser
from loguru import logger

from limb.recording.trigger import TriggerSignal
from limb.tui import SessionState


@dataclass
class ViserSessionPanel:
    """Viser GUI panel for control loop status and data collection.

    Implements the same interface as StatusDisplay:
      - start() / stop()
      - update_loop(hz, step)
      - update_session(state)

    Also implements TriggerSource protocol:
      - get_signal() -> TriggerSignal | None
      - close()

    So GUI buttons feed signals into the session alongside keyboard/pedal.
    """

    viser_server: viser.ViserServer = field(repr=False)

    # Internal state
    _signal_queue: collections.deque = field(default_factory=lambda: collections.deque(maxlen=16), init=False)
    _recording: bool = field(default=False, init=False)
    _started: bool = field(default=False, init=False)

    # GUI handles (populated in start())
    _hz_handle: object = field(default=None, init=False, repr=False)
    _status_handle: object = field(default=None, init=False, repr=False)
    _episode_handle: object = field(default=None, init=False, repr=False)
    _task_handle: object = field(default=None, init=False, repr=False)
    _controls_handle: object = field(default=None, init=False, repr=False)
    _btn_start_stop: object = field(default=None, init=False, repr=False)
    _btn_success: object = field(default=None, init=False, repr=False)
    _btn_discard: object = field(default=None, init=False, repr=False)
    _btn_quit: object = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Create GUI elements in the viser sidebar."""
        if self._started:
            return
        self._started = True
        gui = self.viser_server.gui

        # ── Status display ──
        with gui.add_folder("Data Collection", expand_by_default=True):
            self._hz_handle = gui.add_text("Loop", initial_value="-- Hz | step 0", disabled=True)
            self._status_handle = gui.add_text("Status", initial_value="IDLE", disabled=True)
            self._episode_handle = gui.add_text("Episode", initial_value="0 / 0", disabled=True)
            self._task_handle = gui.add_text("Task", initial_value="(none)", disabled=True)

            # ── Buttons ──
            self._btn_start_stop = gui.add_button("Start", color="green")
            self._btn_success = gui.add_button("Mark Success", color="blue")
            self._btn_discard = gui.add_button("Discard", color="orange")
            self._btn_quit = gui.add_button("Quit Session", color="red")

            self._controls_handle = gui.add_text("Controls", initial_value="", disabled=True)

        # ── Button callbacks → signal queue ──
        @self._btn_start_stop.on_click
        def _(_event: viser.GuiEvent) -> None:
            self._signal_queue.append(TriggerSignal.START_STOP)
            logger.debug("GUI: START_STOP signal")

        @self._btn_success.on_click
        def _(_event: viser.GuiEvent) -> None:
            self._signal_queue.append(TriggerSignal.SUCCESS)
            logger.debug("GUI: SUCCESS signal")

        @self._btn_discard.on_click
        def _(_event: viser.GuiEvent) -> None:
            self._signal_queue.append(TriggerSignal.DISCARD)
            logger.debug("GUI: DISCARD signal")

        @self._btn_quit.on_click
        def _(_event: viser.GuiEvent) -> None:
            self._signal_queue.append(TriggerSignal.QUIT)
            logger.debug("GUI: QUIT signal")

        logger.info("ViserSessionPanel started on viser GUI")

    def stop(self) -> None:
        """No-op — viser GUI lifetime is managed by the server owner."""

    # ── Display interface (matches StatusDisplay) ──

    def update_loop(self, hz: float, step: int) -> None:
        """Update Hz and step count display."""
        if self._hz_handle is not None:
            self._hz_handle.value = f"{hz:.1f} Hz | step {step}"

    def update_session(self, state: SessionState) -> None:
        """Update all session info and button states."""
        if not self._started:
            return

        # Status line: COUNTDOWN / RECORDING / IDLE
        if state.countdown_remaining > 0:
            status = f"COUNTDOWN {state.countdown_remaining:.0f}s"
        elif state.recording:
            status = f"RECORDING {state.episode_duration_s:.0f}s"
        else:
            status = "IDLE"
        if self._status_handle is not None:
            self._status_handle.value = status

        # Episode progress
        target = str(state.episode_total) if state.episode_total > 0 else "\u221e"
        ep_text = f"{state.episode_current} / {target}"
        if state.episodes_successful > 0:
            ep_text += f"  ({state.episodes_successful} success)"
        if state.episodes_discarded > 0:
            ep_text += f"  ({state.episodes_discarded} discarded)"
        if self._episode_handle is not None:
            self._episode_handle.value = ep_text

        # Task
        if self._task_handle is not None:
            task_display = state.task_instruction or "(none)"
            if len(task_display) > 60:
                task_display = task_display[:57] + "..."
            self._task_handle.value = task_display

        # Controls hint
        if self._controls_handle is not None and state.controls_hint:
            self._controls_handle.value = state.controls_hint

        # Toggle start/stop button appearance
        if state.recording and not self._recording:
            if self._btn_start_stop is not None:
                self._btn_start_stop.name = "Stop"
                self._btn_start_stop.color = "red"
            self._recording = True
        elif not state.recording and self._recording:
            if self._btn_start_stop is not None:
                self._btn_start_stop.name = "Start"
                self._btn_start_stop.color = "green"
            self._recording = False

    # ── TriggerSource protocol ──

    def get_signal(self) -> Optional[TriggerSignal]:
        """Non-blocking poll for a GUI button signal."""
        try:
            return self._signal_queue.popleft()
        except IndexError:
            return None

    def close(self) -> None:
        """No-op — server cleanup is external."""
