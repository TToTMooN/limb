"""Multi-episode data collection session manager.

Orchestrates EpisodeRecorder + TriggerSource into a managed collection
workflow. Called from the control loop via step() — no extra threads needed.

Process/thread model during data collection::

    Process 1 (Portal): Camera 1 (RealSense/OpenCV/ZED)
    Process 2 (Portal): Camera 2
    Process 3 (Portal): Left arm robot (CAN bus driver)
    Process 4 (Portal): Right arm robot (CAN bus driver)
    Process 5 (Portal): Agent (teleop device — GELLO/Viser/VR)
    ─────────────────────────────────────────────────────────
    Main process, main thread:
      └─ control loop @ 100 Hz
           ├─ agent.act(obs)         # Portal RPC call (~ms)
           ├─ recorder.record()      # list append + VideoWriter.write (~1-3ms)
           ├─ session.step()         # trigger poll (non-blocking, ~0ms)
           ├─ env.step(action)       # Portal RPC to robots (~ms)
           └─ monitor.update(obs)    # in-process, viser has own server thread
    ─────────────────────────────────────────────────────────
    Episode save (stop_episode): synchronous between episodes.
    Flushes np.save + np.savez + VideoWriter.release (~100-200ms).
    Control loop pauses briefly; robot holds last commanded position.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from limb.core.observation import Observation
from limb.recording.episode_recorder import EpisodeRecorder
from limb.recording.trigger import CompositeTrigger, FootPedalTrigger, KeyboardTrigger, TriggerSignal, VRButtonTrigger
from limb.tui import SessionState


@dataclass
class DataCollectionSession:
    """Manages a multi-episode data collection session.

    Parameters
    ----------
    recorder : EpisodeRecorder
        Handles per-episode raw data recording.
    trigger : TriggerSource
        Hands-free signal input (keyboard/foot pedal/VR buttons).
    num_episodes : int
        Target number of episodes to collect. 0 = unlimited.
    task_instruction : str
        Language instruction stored in metadata (for VLA training data).
    countdown_s : float
        Seconds to wait before recording starts after START_STOP signal.

    Usage in control loop::

        session = DataCollectionSession(recorder=..., trigger=...)
        while not shutdown:
            action = agent.act(obs)
            if not session.step(obs, action):
                break
            obs = env.step(action)
    """

    recorder: Any = field(default_factory=EpisodeRecorder)
    trigger: Any = field(default_factory=KeyboardTrigger)
    num_episodes: int = 10
    task_instruction: str = ""
    countdown_s: float = 3.0

    display: object = None  # StatusDisplay, set programmatically (not from config)

    def __post_init__(self) -> None:
        self._completed: List[Dict[str, Any]] = []
        self._discarded: int = 0
        self._session_start = time.time()
        self._countdown_start: Optional[float] = None
        self._recording_start: Optional[float] = None
        self._done = False

        # Create session subdirectory: base_dir/task_name_YYYYMMDD_HHMMSS/
        if self.task_instruction:
            task_slug = re.sub(r"[^a-z0-9]+", "_", self.task_instruction.lower()).strip("_")[:50]
        else:
            task_slug = "session"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path(self.recorder.base_dir) / f"{task_slug}_{ts}")
        self.recorder.base_dir = session_dir

        logger.info(
            "Data collection session: target={} episodes, task='{}'",
            self.num_episodes if self.num_episodes > 0 else "unlimited",
            self.task_instruction or "(none)",
        )
        self._controls_hint = self._build_controls_hint()
        logger.info("Controls: {}", self._controls_hint)

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def episodes_completed(self) -> int:
        return len(self._completed)

    @property
    def episodes_successful(self) -> int:
        return sum(1 for ep in self._completed if ep.get("success", False))

    def step(self, obs: Observation, action: Dict[str, Any]) -> bool:
        """Called each control loop iteration. Returns False when session is done."""
        if self._done:
            return False

        # Handle countdown (recording starts after countdown_s)
        if self._countdown_start is not None:
            elapsed = time.time() - self._countdown_start
            remaining = self.countdown_s - elapsed
            if remaining > 0:
                self._push_tui_state(countdown_remaining=remaining)
                return True
            # Countdown finished — start recording
            self._countdown_start = None
            self._recording_start = time.time()
            metadata = {"task_instruction": self.task_instruction}
            self.recorder.start_episode(metadata=metadata)
            logger.info("Recording started (episode {})", self.episodes_completed + 1)

        # Poll trigger (non-blocking, ~0ms)
        signal = self.trigger.get_signal()

        if signal == TriggerSignal.START_STOP:
            if self.recorder.is_recording:
                self._finish_episode(success=False)
            else:
                self._start_countdown()

        elif signal == TriggerSignal.SUCCESS:
            if self.recorder.is_recording:
                self._finish_episode(success=True)
            else:
                logger.warning("Not recording — nothing to mark as success")

        elif signal == TriggerSignal.DISCARD:
            if self.recorder.is_recording:
                self._discard_episode()
            else:
                logger.warning("Not recording — nothing to discard")

        elif signal == TriggerSignal.QUIT:
            if self.recorder.is_recording:
                self._finish_episode(success=False)
            self._done = True
            self._print_summary()
            return False

        # Record if active (pre-step obs paired with action = correct s_t, a_t)
        if self.recorder.is_recording:
            self.recorder.record(obs, action)

        self._push_tui_state()

        # Check if target reached
        if self.num_episodes > 0 and self.episodes_completed >= self.num_episodes:
            self._done = True
            self._print_summary()
            return False

        return True

    def _start_countdown(self) -> None:
        if self.countdown_s > 0:
            logger.info("Get ready... recording in {:.0f}s", self.countdown_s)
            self._countdown_start = time.time()
        else:
            self._recording_start = time.time()
            metadata = {"task_instruction": self.task_instruction}
            self.recorder.start_episode(metadata=metadata)
            logger.info("Recording started (episode {})", self.episodes_completed + 1)

    def _finish_episode(self, *, success: bool) -> None:
        """Stop recording, save episode, update stats."""
        episode_dir = self.recorder.stop_episode()
        if episode_dir is None:
            return

        ep_info = {
            "episode_dir": str(episode_dir),
            "success": success,
            "episode_idx": self.episodes_completed,
        }
        self._completed.append(ep_info)

        if success:
            (episode_dir / "SUCCESS").touch()

        status = "SUCCESS" if success else "saved"
        target = self.num_episodes if self.num_episodes > 0 else "inf"
        logger.info(
            "Episode {} ({}) [{}/{}]",
            self.episodes_completed,
            status,
            self.episodes_completed,
            target,
        )

    def _build_controls_hint(self) -> str:
        """Build a controls hint string based on the trigger type."""
        trigger = self.trigger
        # Unwrap CompositeTrigger to find the primary trigger
        if isinstance(trigger, CompositeTrigger):
            for src in trigger.sources:
                if isinstance(src, (FootPedalTrigger, VRButtonTrigger)):
                    trigger = src
                    break

        if isinstance(trigger, FootPedalTrigger):
            left_sig = trigger.left_signal.lower().replace("_", "/")
            right_sig = trigger.right_signal.lower().replace("_", "/")
            return f"[left pedal] {left_sig}  [right pedal] {right_sig}  [S] save  [Q] quit"
        if isinstance(trigger, VRButtonTrigger):
            return "[B] toggle  [Y] discard"
        return "[SPACE] toggle  [D] discard  [S] save  [Q] quit"

    def _push_tui_state(self, countdown_remaining: float = 0.0) -> None:
        """Push current session state to the TUI display (if attached)."""
        if self.display is None:
            return
        duration = 0.0
        if self.recorder.is_recording and self._recording_start is not None:
            duration = time.time() - self._recording_start
        state = SessionState(
            recording=self.recorder.is_recording,
            episode_current=self.episodes_completed + (1 if self.recorder.is_recording else 0),
            episode_total=self.num_episodes,
            episode_duration_s=duration,
            countdown_remaining=countdown_remaining,
            episodes_successful=self.episodes_successful,
            episodes_discarded=self._discarded,
            task_instruction=self.task_instruction,
            controls_hint=self._controls_hint,
        )
        self.display.update_session(state)

    def _discard_episode(self) -> None:
        """Stop recording and delete the episode data."""
        episode_dir = self.recorder.stop_episode()
        if episode_dir is not None:
            shutil.rmtree(episode_dir, ignore_errors=True)
            self._discarded += 1
            logger.info("Episode discarded (data deleted)")

    def _print_summary(self) -> None:
        duration = time.time() - self._session_start
        logger.info("=" * 50)
        logger.info("Data Collection Session Complete")
        logger.info("  Episodes collected: {}", self.episodes_completed)
        logger.info("  Episodes successful: {}", self.episodes_successful)
        logger.info("  Episodes discarded: {}", self._discarded)
        logger.info("  Session duration: {:.1f}s", duration)
        if self._completed:
            logger.info("  Saved to: {}", Path(self._completed[0]["episode_dir"]).parent)
        logger.info("=" * 50)

        # Save session summary JSON alongside episode dirs
        if self._completed:
            summary_dir = Path(self._completed[0]["episode_dir"]).parent
            summary = {
                "task_instruction": self.task_instruction,
                "num_collected": self.episodes_completed,
                "num_successful": self.episodes_successful,
                "num_discarded": self._discarded,
                "duration_s": duration,
                "episodes": self._completed,
            }
            summary_path = summary_dir / "session_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info("Session summary -> {}", summary_path)

    def close(self) -> None:
        """Clean up: save any in-progress episode, close trigger."""
        if self.recorder.is_recording:
            self._finish_episode(success=False)
        self._print_summary()
        self.trigger.close()
        self.recorder.close()
