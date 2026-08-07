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
from limb.recording.trigger import TriggerSignal
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
    trigger : TriggerSource | None
        Optional session-level trigger (e.g. ``KeyboardTrigger``) that
        controls episode start/stop independently of the DAgger pedal
        phase trigger.  When set:
          * the first episode does NOT auto-start — operator must send
            ``START_STOP`` (e.g. press ``SPACE``) to begin episode 1;
          * ``SUCCESS`` while recording  → save episode with ``SUCCESS``
            marker, then wait between episodes for next ``START_STOP``;
          * ``START_STOP`` while recording → save with ``FAILURE``
            marker (policy didn't complete the task), then wait;
          * ``DISCARD`` while recording → throw away the episode, then
            wait;
          * ``QUIT`` → end the session.
          * ``episode_duration_s`` still rotates as a safety cap; on
            timeout the episode is saved as FAILURE and the session
            enters the wait state (operator must press START_STOP for
            the next episode).
        When ``trigger=None`` (default), the legacy time-only rotation
        behaviour is preserved: the first episode auto-starts and each
        rotation auto-starts the next episode marked as success.
    """

    recorder: Any = field(default_factory=EpisodeRecorder)
    num_episodes: int = 0
    task_instruction: str = ""
    # NOTE: corrections-only mode was removed in favor of always-continuous
    # recording. Filtering to correction-only frames is now a one-line
    # `df[df.phase == 'correcting']` at training time, which means no policy
    # success/failure trajectory ever gets thrown away at recording time.
    # The legacy ``record_autonomous`` and ``min_correction_steps`` knobs are
    # kept as fields (so old YAMLs still parse) but no longer influence
    # behaviour — both modes collapse to the previous "sentry" path.
    record_autonomous: bool = True
    episode_duration_s: float = 300.0
    min_correction_steps: int = 0
    trigger: Any = None

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

        # True when an episode has just finished and we're waiting for the
        # operator to reset the scene + press START_STOP to begin the next one.
        # Only used when ``trigger`` is set; in legacy time-only mode this
        # stays False forever (auto-rotation keeps the recorder live).
        self._between_episodes = False

        # Move the recorder into a per-session subdirectory so episodes
        # land neatly under base_dir/<task>_<ts>/.
        if self.task_instruction:
            task_slug = re.sub(r"[^a-z0-9]+", "_", self.task_instruction.lower()).strip("_")[:50]
        else:
            task_slug = "dagger_session"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path(self.recorder.base_dir) / f"{task_slug}_{ts}")
        self.recorder.base_dir = session_dir

        if not self.record_autonomous:
            logger.warning(
                "DAggerCollectionSession: record_autonomous=False is deprecated "
                "(corrections-only mode was removed). Recording continuously; "
                "filter to corrections at training time via phase == 'correcting'."
            )
        logger.info(
            "DAggerCollectionSession: continuous mode, target={} episodes, task='{}'",
            self.num_episodes if self.num_episodes > 0 else "unlimited",
            self.task_instruction or "(none)",
        )

        if self.trigger is None:
            # Legacy path: auto-start the first episode so we don't miss
            # frames while waiting for the first phase transition.
            self._start_episode()
        else:
            # Operator-controlled lifecycle. Wait for the first START_STOP
            # so the operator can stage the scene before recording begins.
            self._between_episodes = True
            logger.info(
                "Press SPACE / Enter (START_STOP) to begin episode 1.  "
                "[S] success+save  [SPACE] failure+save / start next  [D] discard  [Q] quit"
            )

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

        # ---- Session-level trigger (operator lifecycle control) ----
        if self.trigger is not None:
            signal = self.trigger.get_signal()
            if signal == TriggerSignal.SUCCESS:
                if self.recorder.is_recording:
                    self._finish_episode(success=True)
                    self._between_episodes = True
                    logger.info("Episode marked SUCCESS. Reset the scene; press SPACE for the next episode.")
                else:
                    logger.warning("Not recording — nothing to mark as success.")
            elif signal == TriggerSignal.START_STOP:
                if self.recorder.is_recording:
                    # Stop early because the policy didn't finish the task.
                    self._finish_episode(success=False)
                    self._between_episodes = True
                    logger.info("Episode marked FAILURE. Reset the scene; press SPACE for the next episode.")
                else:
                    self._between_episodes = False
                    self._start_episode()
            elif signal == TriggerSignal.DISCARD:
                if self.recorder.is_recording:
                    self._discard_episode()
                    self._between_episodes = True
                    logger.info("Episode discarded. Press SPACE to start the next one.")
                else:
                    logger.warning("Not recording — nothing to discard.")
            elif signal == TriggerSignal.QUIT:
                if self.recorder.is_recording:
                    self._finish_episode(success=False)
                self._done = True
                self._print_summary()
                return False

        # While between episodes the recorder stays idle; the control loop
        # keeps ticking so the agent (and DAgger pedals) remain live.
        if self._between_episodes:
            self._prev_intervention = intervention
            self._push_tui_state(intervention=intervention)
            return True

        # Single recording path: always-continuous (sentry-style). Corrections
        # are labeled per-frame via the recorder's phase metadata so any
        # downstream filter can recover the corrections-only subset.
        self._step_sentry(obs, action, intervention=intervention)

        self._prev_intervention = intervention
        self._push_tui_state(intervention=intervention)

        if self.num_episodes > 0 and self.episodes_completed >= self.num_episodes:
            self._done = True
            # Defensive flush of anything still in flight. In trigger mode
            # this is almost always already saved; in legacy mode it captures
            # the final rotation. Default to FAILURE so an unmarked flush
            # never silently claims success.
            if self.recorder.is_recording:
                self._finish_episode(success=False)
            self._print_summary()
            return False
        return True

    def close(self) -> None:
        # Flush any in-progress episode so we don't leak data on shutdown.
        # Default to FAILURE: a shutdown mid-episode means the operator
        # never got to label it, and silently claiming success would
        # poison training data.
        if self.recorder.is_recording:
            self._finish_episode(success=False)
        if not self._done:
            self._print_summary()
        if self.trigger is not None:
            try:
                self.trigger.close()
            except Exception as e:
                logger.warning("Error closing trigger: {}", e)

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
            if self.trigger is not None:
                # Trigger-controlled mode: the duration cap is a safety net.
                # The operator didn't label this episode within the cap, so
                # we mark FAILURE and hand control back to them rather than
                # silently auto-starting another episode.
                logger.warning(
                    "Episode hit duration cap ({:.1f}s) without an operator label — "
                    "saving as FAILURE. Press SPACE for the next episode.",
                    self.episode_duration_s,
                )
                self._finish_episode(success=False)
                self._between_episodes = True
                return
            logger.info(
                "Rotating episode (elapsed={:.1f}s, episode_duration_s={:.1f}s)", elapsed, self.episode_duration_s
            )
            # Legacy time-only mode: rotation implies success and auto-start
            # of the next episode so the recorder never goes idle.
            self._finish_episode(success=True)
            # Start the next one immediately so the next tick has somewhere to
            # write — but only if we still have episodes to collect.  Otherwise
            # the outer `step()` guard would flush a zero-step episode and we'd
            # overshoot the requested count.
            if self.num_episodes == 0 or self.episodes_completed < self.num_episodes:
                self._start_episode()

    # ------------------------------------------------------------------ #
    #  Episode lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def _start_episode(self) -> None:
        metadata = {
            "task_instruction": self.task_instruction,
            "dagger_mode": "continuous",
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
        (episode_dir / ("SUCCESS" if success else "FAILURE")).touch()

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
        n_fail = self.episodes_completed - self.episodes_successful
        logger.info("=" * 50)
        logger.info("DAgger Collection Session Complete")
        logger.info("  Mode: {}", "trigger" if self.trigger is not None else "time-rotation")
        logger.info("  Episodes collected: {}", self.episodes_completed)
        logger.info("  Episodes successful: {}", self.episodes_successful)
        logger.info("  Episodes failed:     {}", n_fail)
        logger.info("  Episodes discarded:  {}", self._discarded)
        logger.info("  Session duration: {:.1f}s", duration)
        if self._completed:
            session_dir = Path(self._completed[0]["episode_dir"]).parent
            logger.info("  Saved to: {}", session_dir)
            summary = {
                "mode": "continuous",
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
        controls = "[L pedal] pause/resume  [R pedal] correction" + (
            "  [S] success  [SPACE] failure  [D] discard  [Q] quit" if self.trigger is not None else ""
        )
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
