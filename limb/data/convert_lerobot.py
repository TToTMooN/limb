"""Convert limb raw recordings to LeRobot v2.1 dataset format.

No lerobot dependency required — only uses pyarrow and standard lib.

Usage:
    uv run limb convert-lerobot --input-dir recordings/task --output-dir datasets/task
    uv run scripts/data/convert_to_lerobot.py --input-dir recordings/task --output-dir datasets/task

LeRobot v2.1 output structure::

    datasets/task/
      meta/
        info.json
        stats.json
        episodes.jsonl
        tasks.jsonl
      data/
        chunk-000/
          episode_000000.parquet
      videos/
        observation.images.left_wrist_camera/
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

from limb.data.episode_utils import (
    build_action_names,
    build_action_vector,
    build_state_names,
    build_state_vector,
    compute_stats,
    find_episodes,
    load_episode,
)


@dataclass
class Args:
    input_dir: str
    output_dir: str
    task: Optional[str] = None
    robot_type: str = "yam"
    fps: int = 30
    success_only: bool = False
    push_to_hub: Optional[str] = None


def main(args: Args) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow not installed. Run: uv add pyarrow")
        raise SystemExit(1) from None

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    episodes = find_episodes(input_dir, args.success_only)
    if not episodes:
        logger.error("No episodes found in {}", input_dir)
        raise SystemExit(1)

    logger.info("Found {} episodes in {}", len(episodes), input_dir)

    first_ep = load_episode(episodes[0])
    arm_names = sorted(first_ep["arms"].keys())
    cam_names = [c["name"] for c in first_ep["cameras"]]
    task = args.task or first_ep["metadata"].get("task_instruction", "")

    state_names = build_state_names(arm_names, first_ep)
    action_names = build_action_names(arm_names, first_ep)
    state_dim = len(state_names)
    action_dim = len(action_names)

    logger.info("Arms: {}, Cameras: {}", arm_names, cam_names)
    logger.info("State dim: {} ({})", state_dim, state_names)
    logger.info("Action dim: {} ({})", action_dim, action_names)

    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for cam_name in cam_names:
        video_dir = output_dir / "videos" / f"observation.images.{cam_name}" / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)

    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episodes_meta: List[Dict] = []
    total_frames = 0

    for ep_idx, ep_path in enumerate(episodes):
        episode = load_episode(ep_path)
        n_steps = len(episode["timestamps"]) if episode["timestamps"] is not None else 0

        if n_steps == 0:
            logger.warning("Skipping empty episode: {}", ep_path.name)
            continue

        states = build_state_vector(episode, arm_names)
        actions = build_action_vector(episode, arm_names)
        n_steps = min(len(states), len(actions)) if len(actions) > 0 else len(states)
        states = states[:n_steps]
        actions = actions[:n_steps] if len(actions) > 0 else np.zeros((n_steps, action_dim), dtype=np.float32)

        timestamps = (
            episode["timestamps"][:n_steps]
            if episode["timestamps"] is not None
            else np.arange(n_steps, dtype=np.float64) / args.fps
        )
        rel_timestamps = (timestamps - timestamps[0]).astype(np.float32)

        all_states.append(states)
        all_actions.append(actions)

        table_data = {
            "index": pa.array(np.arange(total_frames, total_frames + n_steps, dtype=np.int64)),
            "episode_index": pa.array(np.full(n_steps, ep_idx, dtype=np.int64)),
            "frame_index": pa.array(np.arange(n_steps, dtype=np.int64)),
            "timestamp": pa.array(rel_timestamps),
            "task_index": pa.array(np.zeros(n_steps, dtype=np.int64)),
        }

        for i in range(state_dim):
            table_data[f"observation.state.{i}"] = pa.array(states[:, i])
        for i in range(action_dim):
            table_data[f"action.{i}"] = pa.array(actions[:, i])

        table = pa.table(table_data)
        pq.write_table(table, str(data_dir / f"episode_{ep_idx:06d}.parquet"))

        for cam in episode["cameras"]:
            src = cam["video_path"]
            dst = (
                output_dir / "videos" / f"observation.images.{cam['name']}" / "chunk-000" / f"episode_{ep_idx:06d}.mp4"
            )
            shutil.copy2(str(src), str(dst))

        episodes_meta.append({"episode_index": ep_idx, "tasks": [task], "length": n_steps})
        total_frames += n_steps
        logger.info("  Episode {}: {} steps from {}", ep_idx, n_steps, ep_path.name)

    # Write metadata files
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    stats = compute_stats(all_states, all_actions)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": state_names},
        "action": {"dtype": "float32", "shape": [action_dim], "names": action_names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for cam_name in cam_names:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
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
    logger.info("  Episodes: {}, Total frames: {}", len(episodes_meta), total_frames)
    logger.info("  State dim: {}, Action dim: {}", state_dim, action_dim)
    logger.info("  Cameras: {}", cam_names)

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
    api.upload_folder(folder_path=str(dataset_dir), repo_id=repo_id, repo_type="dataset")
    logger.info("Uploaded: https://huggingface.co/datasets/{}", repo_id)


if __name__ == "__main__":
    main(tyro.cli(Args))
