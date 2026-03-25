"""Shared episode loading utilities for conversion scripts.

Provides episode discovery, loading, and state/action vector building
used by all dataset converters (LeRobot, WebDataset, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger


def find_episodes(input_dir: Path, success_only: bool = False) -> List[Path]:
    """Find valid episode directories, sorted by name.

    Skips:
      - Incomplete episodes (have RECORDING_IN_PROGRESS marker)
      - Episodes without metadata.json (corrupted/interrupted)

    When success_only=True, only includes episodes with a SUCCESS marker.
    """
    episodes = sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))

    valid = []
    for ep in episodes:
        if (ep / "RECORDING_IN_PROGRESS").exists():
            logger.warning("Skipping incomplete episode: {}", ep.name)
            continue
        if not (ep / "metadata.json").exists():
            logger.warning("Skipping episode without metadata: {}", ep.name)
            continue
        valid.append(ep)

    if success_only:
        valid = [ep for ep in valid if (ep / "SUCCESS").exists()]

    return valid


def load_episode(episode_dir: Path) -> Dict:
    """Load all data from a single episode directory."""
    data: Dict = {"dir": episode_dir}

    meta_path = episode_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)
    else:
        data["metadata"] = {}

    ts_path = episode_dir / "timestamps.npy"
    data["timestamps"] = np.load(str(ts_path)) if ts_path.exists() else None

    data["arms"] = {}
    for states_path in episode_dir.glob("*_states.npz"):
        arm_name = states_path.stem.replace("_states", "")
        arm_data = {"states": dict(np.load(str(states_path)))}
        actions_path = episode_dir / f"{arm_name}_actions.npz"
        if actions_path.exists():
            arm_data["actions"] = dict(np.load(str(actions_path)))
        data["arms"][arm_name] = arm_data

    data["cameras"] = []
    for mp4 in sorted(episode_dir.glob("*.mp4")):
        cam_name = mp4.stem
        # Skip depth videos
        if cam_name.endswith("_depth"):
            continue
        cam_ts_path = episode_dir / f"{cam_name}_timestamps.npy"
        data["cameras"].append(
            {
                "name": cam_name,
                "video_path": mp4,
                "timestamps": np.load(str(cam_ts_path)) if cam_ts_path.exists() else None,
            }
        )

    return data


def build_state_vector(episode: Dict, arm_names: List[str]) -> np.ndarray:
    """Concatenate arm states into a single state vector per timestep.

    Order: [left_joint_pos(6), left_gripper(1), right_joint_pos(6), right_gripper(1)] = 14
    """
    parts = []
    for arm_name in arm_names:
        states = episode["arms"][arm_name]["states"]
        parts.append(states["joint_pos"])
        if "gripper_pos" in states:
            parts.append(states["gripper_pos"])
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_action_vector(episode: Dict, arm_names: List[str]) -> np.ndarray:
    """Concatenate arm actions into a single action vector per timestep.

    Order: [left_pos(7), right_pos(7)] = 14
    """
    parts = []
    for arm_name in arm_names:
        arm = episode["arms"][arm_name]
        if "actions" in arm and "pos" in arm["actions"]:
            parts.append(arm["actions"]["pos"])
    return np.concatenate(parts, axis=1).astype(np.float32) if parts else np.empty((0, 0), dtype=np.float32)


def build_state_names(arm_names: List[str], episode: Dict) -> List[str]:
    """Build human-readable names for state vector dimensions."""
    names = []
    for arm_name in arm_names:
        n_joints = episode["arms"][arm_name]["states"]["joint_pos"].shape[1]
        for j in range(n_joints):
            names.append(f"{arm_name}_joint_{j}")
        if "gripper_pos" in episode["arms"][arm_name]["states"]:
            names.append(f"{arm_name}_gripper")
    return names


def build_action_names(arm_names: List[str], episode: Dict) -> List[str]:
    """Build human-readable names for action vector dimensions."""
    names = []
    for arm_name in arm_names:
        if "actions" in episode["arms"][arm_name] and "pos" in episode["arms"][arm_name]["actions"]:
            n_dims = episode["arms"][arm_name]["actions"]["pos"].shape[1]
            for j in range(n_dims):
                names.append(f"{arm_name}_action_{j}")
    return names


def compute_stats(all_states: List[np.ndarray], all_actions: List[np.ndarray]) -> Dict:
    """Compute per-feature statistics (min, max, mean, std)."""
    stats = {}

    if all_states:
        cat_states = np.concatenate(all_states, axis=0)
        stats["observation.state"] = {
            "min": cat_states.min(axis=0).tolist(),
            "max": cat_states.max(axis=0).tolist(),
            "mean": cat_states.mean(axis=0).tolist(),
            "std": cat_states.std(axis=0).tolist(),
            "count": int(cat_states.shape[0]),
        }

    if all_actions:
        cat_actions = np.concatenate(all_actions, axis=0)
        stats["action"] = {
            "min": cat_actions.min(axis=0).tolist(),
            "max": cat_actions.max(axis=0).tolist(),
            "mean": cat_actions.mean(axis=0).tolist(),
            "std": cat_actions.std(axis=0).tolist(),
            "count": int(cat_actions.shape[0]),
        }

    return stats
