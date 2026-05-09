"""Phase-driven data collection for DAgger.

Two modes, selectable at config time:

* **Corrections-only** (``record_autonomous=False``, default).  An episode
  is recorded *only* while the operator is in CORRECTING.  Edge detection
  on the ``intervention`` flag drives the episode lifecycle:

      PAUSED -> CORRECTING   :  recorder.start_episode()
      CORRECTING -> PAUSED   :  recorder.stop_episode()  (+ saved as success)

  Use this when you only want clean human-demonstration data and intend to
  pair each correction with the policy failure that prompted it.

* **Sentry / continuous** (``record_autonomous=True``).  Recording starts
  on the first step and runs forever, with episodes rotated by elapsed
  time.  Every tick is saved with its current ``intervention`` flag.
  Rotation is deferred while a correction is active so episode boundaries
  always land on a clean autonomous frame.

  Use this when you want a complete autonomous-rollout dataset that
  *also* captures the human interventions in-stream.

Either mode is selected via YAML:

    collection:
      _target_: limb.recording.dagger_session.DAggerCollectionSession
      record_autonomous: false        # or true for sentry
      num_episodes: 10
      task_instruction: "..."
      recorder: { ... }

The session reads ``intervention`` and ``phase`` kwargs from
``launch.py``; it never polls a TriggerSource of its own (the foot-pedal
phase trigger lives inside :class:`DAggerAgent`).
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
from limb.tui import SessionState


@dataclass
class DAggerCollectionSession:
    """Phase-driven data collection for DAgger setups.

    Parameters
    ----------
    recorder : EpisodeRecorder
        Per-episode raw data recorder.  ``base_dir`` is repointed at
        ``base_dir/<task_slug>_<ts>/`` so each session lives in its own
        directory.
    num_episodes : int
        Stop after collecting this many episodes.  ``0`` = unlimited
        (Ctrl+C to stop).
    task_instruction : str
        Language instruction stored in episode metadata.
    record_autonomous : bool
        ``False`` (default) -> corrections-only mode.
        ``True``           -> sentry/continuous mode.
    episode_duration_s : float
        Sentry mode only: target duration of one episode before rotation.
        Default 300 s.  Rotation is deferred while CORRECTING so the
        boundary always lands on an autonomous frame.
    min_correction_steps : int
        Corrections-only mode only: discard correction windows that
        finish in fewer than this many ticks (likely accidental pedal
        taps).  Default 5.
    """

    recorder: Any = field(default_factory=EpisodeRecorder)
    num_episodes: int = 0
    task_instruction: str = ""
    record_autonomous: bool = False
    episode_duration_s: float = 300.0
    min_correction_steps: int = 5

    display: object = None  # StatusDisplay, set by launch.py at runtime

    def __post_init__(self) -> None:
        self._completed: List[Dict[str, Any]] = []
        self._discarded: int = 0
        self._session_start = time.time()
        self._done = False

        # Edge-detection state
        self._prev_intervention = False
        # Per-episode counters (corrections-only: spans one CORRECTING window;
        # sentry: spans one rotated episode).
        self._episode_start_time: Optional[float] = None
        self._episode_step_count = 0
        self._episode_intervention_count = 0
        self._latest_phase: Optional[str] = None

        # Move the recorder into a per-session subdirectory so episodes
        # land neatly under base_dir/<task>_<ts>/.
        if self.task_instruction:
            task_slug = re.sub(r"[^a-z0-9]+", "_", self.task_instruction.lower()).strip("_")[:50]
        else:
            task_slug = "dagger_session"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path(self.recorder.base_dir) / f"{task_slug}_{ts}")
        self.recorder.base_dir = session_dir

        mode = "sentry" if self.record_autonomous else "corrections-only"
        logger.info(
            "DAggerCollectionSession: mode={}, target={} episodes, task='{}'",
            mode,
            self.num_episodes if self.num_episodes > 0 else "unlimited",
            self.task_instruction or "(none)",
        )

        # Sentry: start the first episode immediately so we don't miss frames
        # while waiting for the first transition.  Corrections-only: wait for
        # the first PAUSED->CORRECTING edge.
        if self.record_autonomous:
            self._start_episode()

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def episodes_completed(self) -> int:
        return len(self._completed)

    @property
    def episodes_successful(self) -> int:
        return sum(1 for ep in self._completed if ep.get("success", False))

    def step(
        self,
        obs: Observation,
        action: Dict[str, Any],
        *,
        intervention: bool = False,
        phase: Optional[str] = None,
    ) -> bool:
        """Called each control-loop tick.  Returns False when the session is done."""
        if self._done:
            return False
        self._latest_phase = phase

        if self.record_autonomous:
            self._step_sentry(obs, action, intervention=intervention)
        else:
            self._step_corrections_only(obs, action, intervention=intervention)

        self._prev_intervention = intervention
        self._push_tui_state(intervention=intervention)

        if self.num_episodes > 0 and self.episodes_completed >= self.num_episodes:
            self._done = True
            # Make sure we finish whatever is in flight (sentry).
            if self.recorder.is_recording:
                self._finish_episode(success=True)
            self._print_summary()
            return False
        return True

    def close(self) -> None:
        # Flush any in-progress episode so we don't leak data on shutdown.
        if self.recorder.is_recording:
            self._finish_episode(success=True)
        if not self._done:
            self._print_summary()

    # ------------------------------------------------------------------ #
    #  Mode dispatch                                                     #
    # ------------------------------------------------------------------ #

    def _step_corrections_only(self, obs: Observation, action: Dict[str, Any], *, intervention: bool) -> None:
        """Episode lifecycle gated by the rising/falling edge of ``intervention``."""
        # Rising edge: PAUSED -> CORRECTING
        if intervention and not self._prev_intervention:
            self._start_episode()

        # Falling edge: CORRECTING -> PAUSED
        elif not intervention and self._prev_intervention:
            if self._episode_step_count < self.min_correction_steps:
                logger.info(
                    "Discarding short correction ({} steps < min_correction_steps={})",
                    self._episode_step_count,
                    self.min_correction_steps,
                )
                self._discard_episode()
            else:
                self._finish_episode(success=True)

        # Body: only record while CORRECTING.
        if intervention and self.recorder.is_recording:
            self.recorder.record(obs, action, intervention=True)
            self._episode_step_count += 1
            self._episode_intervention_count += 1

    def _step_sentry(self, obs: Observation, action: Dict[str, Any], *, intervention: bool) -> None:
        """Always-on recording with size/time-based episode rotation."""
        if not self.recorder.is_recording:
            # Should only happen between rotations.  Start a fresh one.
            self._start_episode()

        self.recorder.record(obs, action, intervention=intervention)
        self._episode_step_count += 1
        if intervention:
            self._episode_intervention_count += 1

        # Rotate when the episode has run long enough — but defer if we're
        # mid-correction so the boundary lands on a clean autonomous frame.
        elapsed = time.time() - (self._episode_start_time or time.time())
        if elapsed >= self.episode_duration_s and not intervention:
            logger.info(
                "Rotating episode (elapsed={:.1f}s, episode_duration_s={:.1f}s)", elapsed, self.episode_duration_s
            )
            self._finish_episode(success=True)
            # Start the next one immediately so the next tick has somewhere to write.
            self._start_episode()

    # ------------------------------------------------------------------ #
    #  Episode lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def _start_episode(self) -> None:
        metadata = {
            "task_instruction": self.task_instruction,
            "dagger_mode": "sentry" if self.record_autonomous else "corrections_only",
        }
        self.recorder.start_episode(metadata=metadata)
        self._episode_start_time = time.time()
        self._episode_step_count = 0
        self._episode_intervention_count = 0
        logger.info("Episode started ({})", metadata["dagger_mode"])

    def _finish_episode(self, *, success: bool) -> None:
        episode_dir = self.recorder.stop_episode()
        if episode_dir is None:
            return
        ep_info = {
            "episode_dir": str(episode_dir),
            "success": success,
            "episode_idx": self.episodes_completed,
            "step_count": self._episode_step_count,
            "intervention_count": self._episode_intervention_count,
        }
        self._completed.append(ep_info)
        if success:
            (episode_dir / "SUCCESS").touch()

        target = self.num_episodes if self.num_episodes > 0 else "inf"
        logger.info(
            "Episode {}/{} saved ({} steps, {} intervention frames)",
            self.episodes_completed,
            target,
            self._episode_step_count,
            self._episode_intervention_count,
        )

    def _discard_episode(self) -> None:
        episode_dir = self.recorder.stop_episode()
        if episode_dir is None:
            return
        shutil.rmtree(episode_dir, ignore_errors=True)
        self._discarded += 1
        logger.info("Episode discarded (data deleted)")

    def _print_summary(self) -> None:
        duration = time.time() - self._session_start
        logger.info("=" * 50)
        logger.info("DAgger Collection Session Complete")
        logger.info("  Mode: {}", "sentry" if self.record_autonomous else "corrections-only")
        logger.info("  Episodes collected: {}", self.episodes_completed)
        logger.info("  Episodes successful: {}", self.episodes_successful)
        logger.info("  Episodes discarded: {}", self._discarded)
        logger.info("  Session duration: {:.1f}s", duration)
        if self._completed:
            session_dir = Path(self._completed[0]["episode_dir"]).parent
            logger.info("  Saved to: {}", session_dir)
            summary = {
                "mode": "sentry" if self.record_autonomous else "corrections_only",
                "task_instruction": self.task_instruction,
                "num_episodes": self.episodes_completed,
                "num_successful": self.episodes_successful,
                "num_discarded": self._discarded,
                "duration_s": duration,
                "episodes": self._completed,
            }
            (session_dir / "session_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        logger.info("=" * 50)

    # ------------------------------------------------------------------ #
    #  TUI                                                               #
    # ------------------------------------------------------------------ #

    def _push_tui_state(self, *, intervention: bool) -> None:
        if self.display is None:
            return
        duration = 0.0
        if self.recorder.is_recording and self._episode_start_time is not None:
            duration = time.time() - self._episode_start_time
        rate = self._episode_intervention_count / self._episode_step_count if self._episode_step_count > 0 else 0.0
        controls = "[L pedal] pause/resume   [R pedal] correction"
        state = SessionState(
            recording=self.recorder.is_recording,
            episode_current=self.episodes_completed + (1 if self.recorder.is_recording else 0),
            episode_total=self.num_episodes,
            episode_duration_s=duration,
            countdown_remaining=0.0,
            episodes_successful=self.episodes_successful,
            episodes_discarded=self._discarded,
            task_instruction=self.task_instruction,
            controls_hint=controls,
            dagger_phase=self._latest_phase,
            dagger_intervention=intervention,
            dagger_intervention_count=self._episode_intervention_count,
            dagger_total_ticks=self._episode_step_count,
        )
        self.display.update_session(state)
