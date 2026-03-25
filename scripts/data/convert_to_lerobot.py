"""Convert limb raw recordings to LeRobot v2.1 dataset format.

No lerobot dependency required — only uses pyarrow and standard lib.

Usage:
    # Convert a single session (directory containing episode_* subdirs)
    uv run scripts/data/convert_to_lerobot.py --input_dir recordings/red_cube_task --output_dir datasets/red_cube

    # Only include successful episodes
    uv run scripts/data/convert_to_lerobot.py --input_dir recordings/red_cube_task --output_dir datasets/red_cube --success_only

LeRobot v2.1 output structure::

    datasets/red_cube/
      meta/
        info.json
        stats.json
        episodes.jsonl
        tasks.jsonl
      data/
        chunk-000/
          episode_000000.parquet
          episode_000001.parquet
      videos/
        observation.images.left_wrist_camera/
          episode_000000.mp4
        observation.images.right_wrist_camera/
          episode_000000.mp4
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tyro
from loguru import logger


@dataclass
class Args:
    input_dir: str
    output_dir: str
    task: Optional[str] = None  # override task instruction from metadata
    robot_type: str = "yam"
    fps: int = 30
    success_only: bool = False
    push_to_hub: Optional[str] = None  # HuggingFace repo id, e.g. "username/dataset-name"


def _find_episodes(input_dir: Path, success_only: bool) -> List[Path]:
    """Find episode directories, sorted by name.

    Automatically skips:
      - Incomplete episodes (have RECORDING_IN_PROGRESS marker)
      - Episodes without metadata.json (corrupted/interrupted)

    When success_only=True, only includes episodes with a SUCCESS marker.
    """
    episodes = sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))

    valid = []
    for ep in episodes:
        # Skip incomplete recordings (crashed mid-episode)
        if (ep / "RECORDING_IN_PROGRESS").exists():
            logger.warning("Skipping incomplete episode: {}", ep.name)
            continue
        # Skip episodes without metadata (corrupted)
        if not (ep / "metadata.json").exists():
            logger.warning("Skipping episode without metadata: {}", ep.name)
            continue
        valid.append(ep)

    if success_only:
        valid = [ep for ep in valid if (ep / "SUCCESS").exists()]

    return valid


def _load_episode(episode_dir: Path) -> Dict:
    """Load all data from a single episode directory."""
    data = {"dir": episode_dir}

    # Metadata
    meta_path = episode_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)
    else:
        data["metadata"] = {}

    # Timestamps
    ts_path = episode_dir / "timestamps.npy"
    data["timestamps"] = np.load(str(ts_path)) if ts_path.exists() else None

    # Arm states and actions
    data["arms"] = {}
    for states_path in episode_dir.glob("*_states.npz"):
        arm_name = states_path.stem.replace("_states", "")
        arm_data = {"states": dict(np.load(str(states_path)))}
        actions_path = episode_dir / f"{arm_name}_actions.npz"
        if actions_path.exists():
            arm_data["actions"] = dict(np.load(str(actions_path)))
        data["arms"][arm_name] = arm_data

    # Camera info
    data["cameras"] = []
    for mp4 in sorted(episode_dir.glob("*.mp4")):
        cam_name = mp4.stem
        cam_ts_path = episode_dir / f"{cam_name}_timestamps.npy"
        data["cameras"].append(
            {
                "name": cam_name,
                "video_path": mp4,
                "timestamps": np.load(str(cam_ts_path)) if cam_ts_path.exists() else None,
            }
        )

    return data


def _build_state_vector(episode: Dict, arm_names: List[str]) -> np.ndarray:
    """Concatenate arm states into a single state vector per timestep.

    Order: [left_joint_pos(6), left_gripper(1), right_joint_pos(6), right_gripper(1)] = 14
    """
    parts = []
    for arm_name in arm_names:
        arm = episode["arms"][arm_name]
        states = arm["states"]
        parts.append(states["joint_pos"])  # (N, 6)
        if "gripper_pos" in states:
            parts.append(states["gripper_pos"])  # (N, 1)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _build_action_vector(episode: Dict, arm_names: List[str]) -> np.ndarray:
    """Concatenate arm actions into a single action vector per timestep.

    Order: [left_pos(7), right_pos(7)] = 14
    """
    parts = []
    for arm_name in arm_names:
        arm = episode["arms"][arm_name]
        if "actions" in arm and "pos" in arm["actions"]:
            parts.append(arm["actions"]["pos"])  # (N, 7)
    return np.concatenate(parts, axis=1).astype(np.float32) if parts else np.empty((0, 0), dtype=np.float32)


def _build_state_names(arm_names: List[str], episode: Dict) -> List[str]:
    """Build human-readable names for state vector dimensions."""
    names = []
    for arm_name in arm_names:
        n_joints = episode["arms"][arm_name]["states"]["joint_pos"].shape[1]
        for j in range(n_joints):
            names.append(f"{arm_name}_joint_{j}")
        if "gripper_pos" in episode["arms"][arm_name]["states"]:
            names.append(f"{arm_name}_gripper")
    return names


def _build_action_names(arm_names: List[str], episode: Dict) -> List[str]:
    """Build human-readable names for action vector dimensions."""
    names = []
    for arm_name in arm_names:
        if "actions" in episode["arms"][arm_name] and "pos" in episode["arms"][arm_name]["actions"]:
            n_dims = episode["arms"][arm_name]["actions"]["pos"].shape[1]
            for j in range(n_dims):
                names.append(f"{arm_name}_action_{j}")
    return names


def _compute_stats(all_states: List[np.ndarray], all_actions: List[np.ndarray]) -> Dict:
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


def main(args: Args) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow not installed. Run: uv add pyarrow")
        raise SystemExit(1) from None

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    episodes = _find_episodes(input_dir, args.success_only)
    if not episodes:
        logger.error("No episodes found in {}", input_dir)
        raise SystemExit(1)

    logger.info("Found {} episodes in {}", len(episodes), input_dir)

    # Load first episode to determine structure
    first_ep = _load_episode(episodes[0])
    arm_names = sorted(first_ep["arms"].keys())
    cam_names = [c["name"] for c in first_ep["cameras"]]

    # Determine task instruction
    task = args.task or first_ep["metadata"].get("task_instruction", "")

    # Build feature names
    state_names = _build_state_names(arm_names, first_ep)
    action_names = _build_action_names(arm_names, first_ep)
    state_dim = len(state_names)
    action_dim = len(action_names)

    logger.info("Arms: {}, Cameras: {}", arm_names, cam_names)
    logger.info("State dim: {} ({})", state_dim, state_names)
    logger.info("Action dim: {} ({})", action_dim, action_names)

    # Create output directories
    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for cam_name in cam_names:
        video_dir = output_dir / "videos" / f"observation.images.{cam_name}" / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)

    # Process episodes
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episodes_meta: List[Dict] = []
    total_frames = 0

    for ep_idx, ep_path in enumerate(episodes):
        episode = _load_episode(ep_path)
        n_steps = len(episode["timestamps"]) if episode["timestamps"] is not None else 0

        if n_steps == 0:
            logger.warning("Skipping empty episode: {}", ep_path.name)
            continue

        states = _build_state_vector(episode, arm_names)
        actions = _build_action_vector(episode, arm_names)
        n_steps = min(len(states), len(actions)) if len(actions) > 0 else len(states)
        states = states[:n_steps]
        actions = actions[:n_steps] if len(actions) > 0 else np.zeros((n_steps, action_dim), dtype=np.float32)

        timestamps = (
            episode["timestamps"][:n_steps]
            if episode["timestamps"] is not None
            else np.arange(n_steps, dtype=np.float64) / args.fps
        )

        # Relative timestamps (from episode start)
        rel_timestamps = (timestamps - timestamps[0]).astype(np.float32)

        all_states.append(states)
        all_actions.append(actions)

        # Build parquet table for this episode
        table_data = {
            "index": pa.array(np.arange(total_frames, total_frames + n_steps, dtype=np.int64)),
            "episode_index": pa.array(np.full(n_steps, ep_idx, dtype=np.int64)),
            "frame_index": pa.array(np.arange(n_steps, dtype=np.int64)),
            "timestamp": pa.array(rel_timestamps),
            "task_index": pa.array(np.zeros(n_steps, dtype=np.int64)),
        }

        # Add state columns (one column per dimension for LeRobot compat)
        for i in range(state_dim):
            table_data[f"observation.state.{i}"] = pa.array(states[:, i])

        # Add action columns
        for i in range(action_dim):
            table_data[f"action.{i}"] = pa.array(actions[:, i])

        table = pa.table(table_data)
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, str(parquet_path))

        # Copy videos
        for cam in episode["cameras"]:
            src = cam["video_path"]
            dst = (
                output_dir / "videos" / f"observation.images.{cam['name']}" / "chunk-000" / f"episode_{ep_idx:06d}.mp4"
            )
            shutil.copy2(str(src), str(dst))

        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [task],
                "length": n_steps,
            }
        )
        total_frames += n_steps
        logger.info("  Episode {}: {} steps from {}", ep_idx, n_steps, ep_path.name)

    # Write meta/tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")

    # Write meta/episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # Write meta/stats.json
    stats = _compute_stats(all_states, all_actions)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Write meta/info.json
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": action_names,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for cam_name in cam_names:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],  # placeholder, actual dims in video
            "names": ["height", "width", "channels"],
            "info": {"video.fps": args.fps, "video.codec": "mp4v"},
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": 1,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
        "features": features,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info("=" * 50)
    logger.info("LeRobot dataset written to: {}", output_dir)
    logger.info("  Episodes: {}", len(episodes_meta))
    logger.info("  Total frames: {}", total_frames)
    logger.info("  State dim: {}", state_dim)
    logger.info("  Action dim: {}", action_dim)
    logger.info("  Cameras: {}", cam_names)

    # Upload to HuggingFace Hub
    if args.push_to_hub:
        _push_to_hub(output_dir, args.push_to_hub)


def _push_to_hub(dataset_dir: Path, repo_id: str) -> None:
    """Upload dataset to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed. Run: uv pip install huggingface-hub")
        raise SystemExit(1) from None

    api = HfApi()
    logger.info("Uploading to HuggingFace Hub: {}", repo_id)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Uploaded: https://huggingface.co/datasets/{}", repo_id)


if __name__ == "__main__":
    main(tyro.cli(Args))
