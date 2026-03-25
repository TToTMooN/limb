"""Session panel — data collection controls and status in the viser sidebar.

Implements both the display interface (update_loop, update_session) and
TriggerSource protocol (get_signal, close) so GUI buttons feed signals
into the session alongside keyboard/pedal triggers.
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
class SessionPanel:
    """Viser GUI panel for data collection session management.

    Display interface (same as StatusDisplay):
      - update_loop(hz, step)
      - update_session(state)

    TriggerSource protocol:
      - get_signal() -> TriggerSignal | None
      - close()

    ViserPanel protocol:
      - attach(server)
      - detach()
    """

    _server: Optional[viser.ViserServer] = field(default=None, init=False, repr=False)
    _signal_queue: collections.deque = field(default_factory=lambda: collections.deque(maxlen=16), init=False)
    _recording: bool = field(default=False, init=False)
    _attached: bool = field(default=False, init=False)

    # GUI handles
    _hz_handle: object = field(default=None, init=False, repr=False)
    _status_handle: object = field(default=None, init=False, repr=False)
    _episode_handle: object = field(default=None, init=False, repr=False)
    _task_handle: object = field(default=None, init=False, repr=False)
    _controls_handle: object = field(default=None, init=False, repr=False)
    _btn_start_stop: object = field(default=None, init=False, repr=False)

    def attach(self, server: viser.ViserServer) -> None:
        """Create GUI elements in the viser sidebar."""
        if self._attached:
            return
        self._server = server
        self._attached = True
        gui = server.gui

        with gui.add_folder("Data Collection", expand_by_default=True):
            self._hz_handle = gui.add_text("Loop", initial_value="-- Hz | step 0", disabled=True)
            self._status_handle = gui.add_text("Status", initial_value="IDLE", disabled=True)
            self._episode_handle = gui.add_text("Episode", initial_value="0 / 0", disabled=True)
            self._task_handle = gui.add_text("Task", initial_value="(none)", disabled=True)

            self._btn_start_stop = gui.add_button("Start", color="green")
            btn_success = gui.add_button("Mark Success", color="blue")
            btn_discard = gui.add_button("Discard", color="orange")
            btn_quit = gui.add_button("Quit Session", color="red")

            self._controls_handle = gui.add_text("Controls", initial_value="", disabled=True)

        @self._btn_start_stop.on_click
        def _(_event: viser.GuiEvent) -> None:
            self._signal_queue.append(TriggerSignal.START_STOP)

        @btn_success.on_click
        def _(_event: viser.GuiEvent) -> None:
            self._signal_queue.append(TriggerSignal.SUCCESS)

        @btn_discard.on_click
        def _(_event: viser.GuiEvent) -> None:
            self._signal_queue.append(TriggerSignal.DISCARD)

        @btn_quit.on_click
        def _(_event: viser.GuiEvent) -> None:
            self._signal_queue.append(TriggerSignal.QUIT)

        logger.info("SessionPanel attached to viser GUI")

    # Alias for StatusDisplay compatibility
    def start(self) -> None:
        """No-op — attach() does the work. Kept for StatusDisplay compat."""

    def stop(self) -> None:
        """No-op — detach() does cleanup."""

    # ── Display interface ──

    def update_loop(self, hz: float, step: int) -> None:
        if self._hz_handle is not None:
            self._hz_handle.value = f"{hz:.1f} Hz | step {step}"

    def update_session(self, state: SessionState) -> None:
        if not self._attached:
            return

        # Status
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

        # Toggle start/stop button
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
        try:
            return self._signal_queue.popleft()
        except IndexError:
            return None

    def close(self) -> None:
        pass

    def detach(self) -> None:
        pass
