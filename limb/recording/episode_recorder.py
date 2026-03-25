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

        # Depth recording (when cameras provide depth data)
        self._depth_writers: Dict[str, AsyncVideoWriter] = {}

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
        Inspired by Raiden's automatic override of incomplete recordings.
        """
        import shutil

        base = Path(self.base_dir)
        if not base.exists():
            return

        for ep_dir in sorted(base.glob("episode_*")):
            marker = ep_dir / "RECORDING_IN_PROGRESS"
            metadata = ep_dir / "metadata.json"
            if marker.exists() and not metadata.exists():
                logger.warning("Removing incomplete episode: {}", ep_dir.name)
                shutil.rmtree(ep_dir, ignore_errors=True)
            elif marker.exists() and metadata.exists():
                # Recording finished but marker wasn't cleaned up (crash during save)
                marker.unlink(missing_ok=True)

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
            self._arm_states = {}
            self._actions = {}
            self._writers = {}
            self._cam_timestamps = {}
            self._depth_writers = {}
            self._metadata = metadata or {}
            self._metadata["start_time"] = time.time()
            self._metadata["start_time_str"] = ts
            if self.ee_frame_names is not None:
                self._metadata["ee_frame_names"] = self.ee_frame_names

            # Mark episode as in-progress (removed on successful save)
            (self._episode_dir / "RECORDING_IN_PROGRESS").touch()

            self._recording = True
            self._episode_count += 1
            logger.info("Episode recording started -> {}", self._episode_dir)
            return self._episode_dir

    def record(self, obs: Observation, action: Dict[str, Any]) -> None:
        """Record one timestep of observation + action.

        Args:
            obs: Typed Observation from RobotEnv.
            action: Action dict as returned by agent.act().
        """
        with self._lock:
            if not self._recording:
                return

            self._timestamps.append(obs.timestamp)

            # Record arm states
            for arm_name, arm_obs in obs.arms.items():
                if arm_name not in self._arm_states:
                    self._arm_states[arm_name] = {
                        "joint_pos": [],
                        "joint_vel": [],
                        "gripper_pos": [],
                        "ee_pose": [],
                    }
                self._arm_states[arm_name]["joint_pos"].append(arm_obs.joint_pos.copy())
                self._arm_states[arm_name]["joint_vel"].append(arm_obs.joint_vel.copy())
                if arm_obs.gripper_pos is not None:
                    self._arm_states[arm_name]["gripper_pos"].append(arm_obs.gripper_pos.copy())
                if arm_obs.ee_pose is not None:
                    self._arm_states[arm_name]["ee_pose"].append(arm_obs.ee_pose.copy())

            # Record actions
            for arm_name, arm_action in action.items():
                if not isinstance(arm_action, dict):
                    continue
                if arm_name not in self._actions:
                    self._actions[arm_name] = {}
                if "pos" in arm_action:
                    self._actions[arm_name].setdefault("pos", []).append(np.asarray(arm_action["pos"]))
                if "vel" in arm_action:
                    self._actions[arm_name].setdefault("vel", []).append(np.asarray(arm_action["vel"]))

            # Record camera frames as video (async, NVENC when available)
            for cam_name, cam_obs in obs.cameras.items():
                if cam_name not in self._writers:
                    h, w = cam_obs.rgb.shape[:2]
                    path = str(self._episode_dir / f"{cam_name}.mp4")
                    writer = AsyncVideoWriter(path=path, width=w, height=h, fps=self.recording_fps)
                    writer.start()
                    self._writers[cam_name] = writer
                    self._cam_timestamps[cam_name] = []
                self._writers[cam_name].write(cam_obs.rgb)
                self._cam_timestamps[cam_name].append(cam_obs.timestamp)

                # Record depth as 16-bit grayscale video when available
                depth = getattr(cam_obs, "depth", None)
                if depth is not None and cam_name not in self._depth_writers:
                    h, w = depth.shape[:2]
                    depth_path = str(self._episode_dir / f"{cam_name}_depth.mp4")
                    depth_writer = AsyncVideoWriter(
                        path=depth_path, width=w, height=h, fps=self.recording_fps, pix_fmt="gray16le"
                    )
                    depth_writer.start()
                    self._depth_writers[cam_name] = depth_writer
                if depth is not None and cam_name in self._depth_writers:
                    self._depth_writers[cam_name].write(depth)

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
        for w in self._depth_writers.values():
            w.stop()

        # Save timestamps
        np.save(str(episode_dir / "timestamps.npy"), np.array(self._timestamps, dtype=np.float64))

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

        # Save metadata
        self._metadata["end_time"] = time.time()
        self._metadata["num_steps"] = self._step_idx
        self._metadata["duration_s"] = self._metadata["end_time"] - self._metadata["start_time"]
        self._metadata["recording_fps"] = self.recording_fps
        self._metadata["cameras"] = list(self._cam_timestamps.keys())
        self._metadata["arms"] = list(self._arm_states.keys())
        self._metadata["has_depth"] = bool(self._depth_writers)
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
