"""Episode recorder for capturing raw control loop data.

Records every observation + action at the control loop frequency.
Output structure per episode::

    recordings/episode_20260304_153045_0001/
        metadata.json                       # config, robot info, timing, ee frame names
        timestamps.npy                      # (N,) float64 Unix timestamps
        {arm}_states.npz                    # joint_pos (N,6), joint_vel (N,6), gripper_pos (N,1), ee_pose (N,7)
        {arm}_actions.npz                   # pos (N,7), optionally vel (N,7)
        {camera}.mp4                        # video per camera
        {camera}_timestamps.npy             # (N,) per-frame camera timestamps

Post-processing to HDF5, LeRobot format, etc. is done by separate scripts.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from robocam import AsyncVideoWriter

from limb.core.observation import Observation


@dataclass
class EpisodeRecorder:
    """Records raw episode data at the control loop rate.

    Parameters
    ----------
    base_dir : str
        Root directory for recordings.
    recording_fps : int
        FPS for video encoding (should match or approximate control rate).
    auto_start : bool
        If True, recording begins immediately on creation.
    ee_frame_names : dict or None
        Mapping of arm name to EE frame name, e.g. {"left": "ee_link", "right": "ee_link"}.
        Stored in metadata for downstream processing.
    """

    base_dir: str = "recordings"
    recording_fps: int = 30
    auto_start: bool = False
    ee_frame_names: Optional[Dict[str, str]] = None
    robot_configs: Optional[Dict[str, Any]] = None
    # Video codec passed to robocam.AsyncVideoWriter. "auto" picks hevc_nvenc when the
    # encoder EXISTS — but NVENC also needs free VRAM at runtime: with the pi0.5 serve
    # + SAM3 server resident (2026-07-07 22:47 run) every encoder OOM'd and all three
    # videos saved EMPTY. Set "libx264" (CPU) for GPU-crowded sessions.
    video_codec: str = "auto"

    def __post_init__(self) -> None:
        self._recording = False
        self._episode_dir: Optional[Path] = None
        self._episode_count = self._find_next_episode_count()
        self._step_idx = 0
        self._lock = threading.Lock()

        # Per-step buffers
        self._timestamps: List[float] = []
        self._arm_states: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self._actions: Dict[str, Dict[str, List[np.ndarray]]] = {}

        # Video writers (async ffmpeg-based, NVENC when available)
        self._writers: Dict[str, AsyncVideoWriter] = {}
        self._cam_timestamps: Dict[str, List[float]] = {}

        # Metadata
        self._metadata: Dict[str, Any] = {}

        # Clean up any incomplete episodes from previous runs
        self._cleanup_incomplete_episodes()

        if self.auto_start:
            self.start_episode()

    def _find_next_episode_count(self) -> int:
        """Find the next episode number by scanning existing directories."""
        base = Path(self.base_dir)
        if not base.exists():
            return 0
        existing = sorted(base.glob("episode_*"))
        return len(existing)

    def _cleanup_incomplete_episodes(self) -> None:
        """Remove incomplete episodes from previous runs.

        An episode is considered incomplete if it has a RECORDING_IN_PROGRESS
        marker but no metadata.json (meaning it was interrupted before saving).

        Safety: the marker file contains the PID that created it. We only
        delete an episode if the owning process is no longer running, to
        avoid removing another process's active recording.
        """
        import shutil

        base = Path(self.base_dir)
        if not base.exists():
            return

        for ep_dir in sorted(base.glob("episode_*")):
            marker = ep_dir / "RECORDING_IN_PROGRESS"
            metadata = ep_dir / "metadata.json"
            if marker.exists() and not metadata.exists():
                # Check if the owning process is still alive
                if self._is_marker_owner_alive(marker):
                    logger.debug("Skipping active recording (owner alive): {}", ep_dir.name)
                    continue
                logger.warning("Removing incomplete episode: {}", ep_dir.name)
                shutil.rmtree(ep_dir, ignore_errors=True)
            elif marker.exists() and metadata.exists():
                # Recording finished but marker wasn't cleaned up (crash during save)
                marker.unlink(missing_ok=True)

    @staticmethod
    def _read_proc_starttime(pid: int) -> Optional[int]:
        """Read a process's start time (clock ticks since boot) from /proc/<pid>/stat.

        Parses field 22 of /proc/<pid>/stat (`starttime` per proc(5)). This is
        an immutable per-process identifier — unlike the PID itself, it does
        not get recycled. Used to detect PID reuse when a marker file outlives
        its original owner.
        """
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                data = f.read()
        except OSError:
            return None
        # The `comm` field (2nd) may contain arbitrary bytes including spaces,
        # but is wrapped in parens — find the last ')' to skip past it safely.
        rparen = data.rfind(b")")
        if rparen < 0:
            return None
        try:
            fields = data[rparen + 2 :].split()
            # After (pid comm), the remaining fields are state(0), ppid(1), ...
            # starttime is field 22 in proc(5), i.e. index 19 in this tail slice.
            return int(fields[19])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _is_marker_owner_alive(marker: Path) -> bool:
        """Check if the process that wrote the marker is still running.

        Reads `<pid>\\n<starttime>` from the marker and compares both to the
        current state of /proc. A bare-PID marker (written before start time
        was tracked) is treated as alive iff the PID still exists — slightly
        looser but backwards-compatible.
        """
        try:
            parts = marker.read_text().strip().split("\n")
            pid = int(parts[0])
        except (ValueError, OSError):
            return False

        current_starttime = EpisodeRecorder._read_proc_starttime(pid)
        if current_starttime is None:
            return False

        if len(parts) < 2:
            # Legacy marker without starttime — fall back to PID-exists check.
            return True
        try:
            stored_starttime = int(parts[1])
        except ValueError:
            return True
        return stored_starttime == current_starttime

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start_episode(self, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Begin recording a new episode. Returns the episode directory path."""
        with self._lock:
            if self._recording:
                self._stop_episode_unlocked()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._episode_dir = Path(self.base_dir) / f"episode_{ts}_{self._episode_count:04d}"
            self._episode_dir.mkdir(parents=True, exist_ok=True)

            self._step_idx = 0
            self._timestamps = []
            self._interventions: List[bool] = []
            # RECAP-style phase tracking. Default phase is "autonomous" so a
            # non-DAgger recorder run still produces a valid phases.npy file
            # (all-autonomous) that downstream tooling can rely on existing.
            self._phases: List[str] = []
            self._rewards: List[float] = []
            self._successes: List[bool] = []
            self._correction_indices: List[int] = []
            self._arm_states = {}
            self._actions = {}
            # Per-arm raw policy-target stream (commanded action is in
            # self._actions["pos"]). Differs during the resume-blend window;
            # equal outside it; absent in non-DAgger / teleop runs.
            self._policy_actions: Dict[str, Dict[str, List[np.ndarray]]] = {}
            self._writers = {}
            self._cam_timestamps = {}
            self._metadata = metadata or {}
            self._metadata["start_time"] = time.time()
            self._metadata["start_time_str"] = ts
            if self.ee_frame_names is not None:
                self._metadata["ee_frame_names"] = self.ee_frame_names

            # Mark episode as in-progress with our PID + process start time
            # (immune to PID reuse; removed on successful save).
            import os

            pid = os.getpid()
            starttime = self._read_proc_starttime(pid)
            marker_text = f"{pid}\n{starttime}\n" if starttime is not None else f"{pid}\n"
            (self._episode_dir / "RECORDING_IN_PROGRESS").write_text(marker_text)

            self._recording = True
            self._episode_count += 1
            logger.info("Episode recording started -> {}", self._episode_dir)
            return self._episode_dir

    def record(self, obs: Observation, action: Dict[str, Any], *, intervention: bool = False) -> None:
        """Record one timestep of observation + action.

        Args:
            obs: Typed Observation from RobotEnv.
            action: Action dict as returned by agent.act(). May contain the
                following non-arm metadata keys (consumed by this recorder
                and stripped before any other use):

                - ``_phase``: "autonomous" | "paused" | "correcting"
                - ``_correction_index``: int — increments on each entry into
                  CORRECTING; 0 baseline.
                - per-arm ``policy_pos``: the policy's raw target before the
                  resume-blend window. Outside the blend it equals ``pos``.

                Saved to ``phase.npy``, ``correction_index.npy``, and
                ``{arm}_policy_actions.npz`` respectively.
            intervention: legacy boolean kept for non-DAgger callers. If
                ``_phase`` is present on the action dict it takes precedence.
        """
        with self._lock:
            if not self._recording:
                return

            # Phase metadata: prefer DAggerAgent's stamped values if present.
            phase = action.get("_phase")
            correction_idx = action.get("_correction_index", 0)
            if phase is None:
                phase = "correcting" if intervention else "autonomous"
                derived_intervention = bool(intervention)
            else:
                derived_intervention = (phase == "correcting")

            self._timestamps.append(obs.timestamp)
            self._interventions.append(derived_intervention)
            self._phases.append(str(phase))
            self._correction_indices.append(int(correction_idx))
            # SubRL per-frame verifier labels (user 2026-07-08: the recording must be
            # self-contained for auditing rollout success/failure): the agent stamps
            # _reward/_success on every action; absent (non-SubRL agents) -> 0/False.
            self._rewards.append(float(action.get("_reward", 0.0) or 0.0))
            self._successes.append(bool(action.get("_success", False)))

            # Record arm states
            for arm_name, arm_obs in obs.arms.items():
                if arm_name not in self._arm_states:
                    self._arm_states[arm_name] = {
                        "joint_pos": [],
                        "joint_vel": [],
                        "gripper_pos": [],
                        "ee_pose": [],
                        "joint_eff": [],
                    }
                self._arm_states[arm_name]["joint_pos"].append(arm_obs.joint_pos.copy())
                self._arm_states[arm_name]["joint_vel"].append(arm_obs.joint_vel.copy())
                if arm_obs.gripper_pos is not None:
                    self._arm_states[arm_name]["gripper_pos"].append(arm_obs.gripper_pos.copy())
                if arm_obs.ee_pose is not None:
                    self._arm_states[arm_name]["ee_pose"].append(arm_obs.ee_pose.copy())
                if arm_obs.joint_eff is not None:
                    # motor-current efforts — needed to calibrate the SubRL gripper-effort
                    # grasp thresholds (LOAD_LOW/LOAD_HOLD) from recorded grasp episodes
                    self._arm_states[arm_name]["joint_eff"].append(arm_obs.joint_eff.copy())

            # Record actions (and the policy_pos shadow stream when present).
            # For composite agents like DAggerAgent the action dict can be {}
            # (PAUSED) or omit specific arms (leaders during CORRECTING). The
            # physical robot holds the last commanded position via PD in those
            # cases, so we record the same thing — the previous tick's value —
            # so every action array stays aligned with timestamps / states.
            for arm_name, arm_action in action.items():
                if isinstance(arm_name, str) and arm_name.startswith("_"):
                    continue  # phase metadata handled above
                if not isinstance(arm_action, dict):
                    continue
                if arm_name not in self._actions:
                    self._actions[arm_name] = {}
                if "pos" in arm_action:
                    self._actions[arm_name].setdefault("pos", []).append(np.asarray(arm_action["pos"]))
                if "vel" in arm_action:
                    self._actions[arm_name].setdefault("vel", []).append(np.asarray(arm_action["vel"]))
                if "policy_pos" in arm_action:
                    self._policy_actions.setdefault(arm_name, {}).setdefault(
                        "pos", []
                    ).append(np.asarray(arm_action["policy_pos"]))

            # Hold-the-last-value padding for arms that were silent this tick.
            # The physical command was "hold last commanded position" (PD on
            # the env side), so the recorded action stream stays causally
            # consistent with what the robot actually did.
            target_len = self._step_idx + 1

            def _pad_to_target(store: Dict[str, Dict[str, List[np.ndarray]]]) -> None:
                for arm_store in store.values():
                    for history in arm_store.values():
                        while len(history) < target_len:
                            if history:
                                history.append(history[-1].copy())
                            else:
                                break  # arm never seen yet — leave the gap

            _pad_to_target(self._actions)
            _pad_to_target(self._policy_actions)

            # Record camera frames as video (async, NVENC when available)
            for cam_name, cam_obs in obs.cameras.items():
                if cam_name not in self._writers:
                    h, w = cam_obs.rgb.shape[:2]
                    path = str(self._episode_dir / f"{cam_name}.mp4")
                    writer = AsyncVideoWriter(path=path, width=w, height=h,
                                              fps=self.recording_fps, codec=self.video_codec)
                    # Spawn ffmpeg DETACHED from the terminal session: Ctrl+C is
                    # delivered to the whole foreground group, so the encoders got
                    # SIGINT ("Exiting normally, received signal 2", exit 255) before
                    # the recorder's graceful stop() — harmless for the files but a
                    # spurious WARNING per camera on every shutdown (23:40 run).
                    # setpgid-after-exec is EPERM, so inject start_new_session at
                    # spawn via a scoped patch of robocam's Popen.
                    import robocam.video_writer as _rvw
                    _orig_popen = _rvw.subprocess.Popen

                    def _detached_popen(*a, _orig_popen=_orig_popen, **kw):
                        kw.setdefault("start_new_session", True)
                        # NEVER pipe ffmpeg's stderr without a reader (on-robot freezes
                        # 2026-07-08, both at ~366 s / ~8350 frames): robocam passes
                        # stderr=PIPE and nothing drains it, so ffmpeg's progress lines
                        # fill the 64 KB pipe -> ffmpeg blocks -> stops reading stdin ->
                        # writer queue (300) fills -> write() blocks -> the whole 30 Hz
                        # control loop hangs. Encoder crashes still surface as
                        # BrokenPipeError on stdin (robocam logs it and sets _failed).
                        kw["stderr"] = _rvw.subprocess.DEVNULL
                        return _orig_popen(*a, **kw)

                    _rvw.subprocess.Popen = _detached_popen
                    try:
                        writer.start()
                    finally:
                        _rvw.subprocess.Popen = _orig_popen
                    self._writers[cam_name] = writer
                    self._cam_timestamps[cam_name] = []
                # DROP the frame rather than block the control loop when the encoder
                # falls behind (robocam's write() is a BLOCKING queue.put once its
                # 300-frame buffer is full — the 2026-07-08 freeze amplifier).
                w_ = self._writers[cam_name]
                if getattr(w_, "_queue", None) is not None and w_._queue.full():
                    # timestamp intentionally NOT appended: <cam>_timestamps.npy must
                    # stay 1:1 with the frames actually encoded into the mp4.
                    now = time.time()
                    if now - getattr(self, "_drop_warn_t", 0.0) > 5.0:
                        self._drop_warn_t = now
                        logger.warning(f"video writer '{cam_name}' queue full — dropping "
                                       "frames instead of stalling the control loop")
                else:
                    w_.write(cam_obs.rgb)
                    self._cam_timestamps[cam_name].append(cam_obs.timestamp)

                # NOTE: depth recording is intentionally not implemented yet.
                # robocam.AsyncVideoWriter is hardcoded for 8-bit RGB input; a
                # correct 16-bit depth pipeline (FFV1 or compressed npz) should
                # live in robocam, not here. Track in robocam when needed.

            self._step_idx += 1

    def stop_episode(self) -> Optional[Path]:
        """Finish the current episode, flush all data to disk."""
        with self._lock:
            return self._stop_episode_unlocked()

    def _stop_episode_unlocked(self) -> Optional[Path]:
        """Internal stop — caller must hold self._lock."""
        if not self._recording:
            return None

        episode_dir = self._episode_dir
        self._recording = False

        # Flush async video writers (waits for ffmpeg to finish)
        for w in self._writers.values():
            w.stop()

        # Save timestamps
        np.save(str(episode_dir / "timestamps.npy"), np.array(self._timestamps, dtype=np.float64))

        # Save per-tick intervention flags (DAgger): True iff frame was an
        # operator correction.  Always written so episode shape is uniform
        # regardless of whether the agent supports interventions.
        interventions_arr = np.asarray(self._interventions, dtype=bool)
        np.save(str(episode_dir / "interventions.npy"), interventions_arr)

        # Richer phase + correction tracking (RECAP-style). phase.npy is the
        # categorical equivalent of interventions.npy with PAUSED distinguished
        # from AUTONOMOUS; correction_index.npy lets downstream tools group
        # frames into discrete correction episodes.
        if self._phases:
            np.save(
                str(episode_dir / "phase.npy"),
                np.asarray(self._phases, dtype=np.dtype("U16")),
            )
        if self._rewards:
            # reward.npy / success.npy: per-frame verifier verdicts (SubRL) — a
            # success rollout shows success=True + reward 1.0 on its terminal frames.
            np.save(str(episode_dir / "reward.npy"),
                    np.asarray(self._rewards, dtype=np.float32))
            np.save(str(episode_dir / "success.npy"),
                    np.asarray(self._successes, dtype=bool))
        if self._correction_indices:
            np.save(
                str(episode_dir / "correction_index.npy"),
                np.asarray(self._correction_indices, dtype=np.int32),
            )

        # Save per-camera timestamps
        for cam_name, cam_ts in self._cam_timestamps.items():
            np.save(str(episode_dir / f"{cam_name}_timestamps.npy"), np.array(cam_ts, dtype=np.float64))

        # Save arm states as npz
        for arm_name, state_dict in self._arm_states.items():
            arrays = {}
            for key, val_list in state_dict.items():
                if val_list:
                    arrays[key] = np.stack(val_list)
            if arrays:
                np.savez(str(episode_dir / f"{arm_name}_states.npz"), **arrays)

        # Save actions as npz
        for arm_name, action_dict in self._actions.items():
            arrays = {}
            for key, val_list in action_dict.items():
                if val_list:
                    arrays[key] = np.stack(val_list)
            if arrays:
                np.savez(str(episode_dir / f"{arm_name}_actions.npz"), **arrays)

        # Save the policy_pos shadow stream (DAgger / composite agents only;
        # files are skipped when empty so non-DAgger recordings stay clean).
        for arm_name, action_dict in self._policy_actions.items():
            arrays = {}
            for key, val_list in action_dict.items():
                if val_list:
                    arrays[key] = np.stack(val_list)
            if arrays:
                np.savez(str(episode_dir / f"{arm_name}_policy_actions.npz"), **arrays)

        # Save metadata
        self._metadata["end_time"] = time.time()
        self._metadata["num_steps"] = self._step_idx
        self._metadata["duration_s"] = self._metadata["end_time"] - self._metadata["start_time"]
        self._metadata["recording_fps"] = self.recording_fps
        self._metadata["cameras"] = list(self._cam_timestamps.keys())
        self._metadata["arms"] = list(self._arm_states.keys())
        intervention_count = int(interventions_arr.sum())
        self._metadata["intervention_count"] = intervention_count
        self._metadata["intervention_fraction"] = (
            intervention_count / len(interventions_arr) if len(interventions_arr) > 0 else 0.0
        )
        if self.robot_configs is not None:
            self._metadata["robot_configs"] = self.robot_configs
        with open(str(episode_dir / "metadata.json"), "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

        # Remove in-progress marker — episode is now complete
        (episode_dir / "RECORDING_IN_PROGRESS").unlink(missing_ok=True)

        logger.info(
            "Episode saved: {} ({} steps, {:.1f}s)",
            episode_dir,
            self._step_idx,
            self._metadata["duration_s"],
        )
        return episode_dir

    def close(self) -> None:
        """Stop recording and clean up."""
        if self._recording:
            self.stop_episode()
