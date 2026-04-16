"""Convert limb raw recordings to LeRobot v3.0 dataset format.

No lerobot dependency required — only uses pyarrow and standard lib.

Usage:
    uv run limb convert-lerobot --input-dir recordings/task --output-dir datasets/task

LeRobot v3.0 output structure::

    datasets/task/
      meta/
        info.json
        stats.json
        tasks.parquet
        episodes/
          chunk-000/
            file-000.parquet
      data/
        chunk-000/
          file-000.parquet       # may contain 1+ episodes
      videos/
        observation.images.left_wrist_camera/
          chunk-000/
            file-000.mp4
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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

CODEBASE_VERSION = "v3.0"


@dataclass
class Args:
    input_dir: str
    output_dir: str
    task: Optional[str] = None
    robot_type: str = "yam"
    fps: int = 30
    success_only: bool = False
    push_to_hub: Optional[str] = None


def _probe_video_codec(path: Path) -> str:
    """Detect the actual video codec of an mp4 file via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0",
             str(path)],
            capture_output=True, text=True, timeout=5,
        )
        codec = result.stdout.strip()
        if codec:
            return codec
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "hevc"


def _compute_episode_stats(
    states: np.ndarray,
    actions: np.ndarray,
    cam_names: List[str],
    episode_length: int,
    fps: int,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-feature stats for a single episode (v3.0 episodes parquet)."""
    stats: Dict[str, Dict[str, Any]] = {}

    def _numeric_stats(arr: np.ndarray, dtype: str = "float") -> Dict[str, Any]:
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }

    stats["observation.state"] = _numeric_stats(states)
    stats["action"] = _numeric_stats(actions)

    n = episode_length
    indices = np.arange(n, dtype=np.float64)
    for key, arr in [
        ("episode_index", np.zeros(n, dtype=np.float64)),
        ("frame_index", indices),
        ("timestamp", indices / fps),
        ("index", indices),
        ("task_index", np.zeros(n, dtype=np.float64)),
    ]:
        stats[key] = _numeric_stats(arr.reshape(-1, 1))

    for cam_name in cam_names:
        stats[f"observation.images.{cam_name}"] = {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[255.0]], [[255.0]], [[255.0]]],
            "mean": [[[128.0]], [[128.0]], [[128.0]]],
            "std": [[[75.0]], [[75.0]], [[75.0]]],
            "count": [n],
        }

    return stats


def main(args: Args) -> None:
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

    # Create directory structure
    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    episodes_meta_dir = meta_dir / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_meta_dir.mkdir(parents=True, exist_ok=True)

    for cam_name in cam_names:
        video_dir = output_dir / "videos" / f"observation.images.{cam_name}" / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)

    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episodes_rows: List[Dict[str, Any]] = []
    total_frames = 0
    video_codec: Optional[str] = None
    total_data_bytes = 0
    total_video_bytes = 0

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

        all_states.append(states)
        all_actions.append(actions)

        # Write data parquet with list<float32> columns
        table_data = {
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
            "episode_index": pa.array(np.full(n_steps, ep_idx, dtype=np.int64)),
            "frame_index": pa.array(np.arange(n_steps, dtype=np.int64)),
            "timestamp": pa.array((np.arange(n_steps, dtype=np.float64) / args.fps).astype(np.float32)),
            "index": pa.array(np.arange(total_frames, total_frames + n_steps, dtype=np.int64)),
            "task_index": pa.array(np.zeros(n_steps, dtype=np.int64)),
        }
        table = pa.table(table_data)
        parquet_path = data_dir / f"file-{ep_idx:03d}.parquet"
        pq.write_table(table, str(parquet_path), compression="snappy")
        total_data_bytes += parquet_path.stat().st_size

        # Copy videos
        for cam in episode["cameras"]:
            src = cam["video_path"]
            dst = output_dir / "videos" / f"observation.images.{cam['name']}" / "chunk-000" / f"file-{ep_idx:03d}.mp4"
            shutil.copy2(str(src), str(dst))
            total_video_bytes += dst.stat().st_size
            if video_codec is None:
                video_codec = _probe_video_codec(dst)

        # Build episodes metadata row
        ep_stats = _compute_episode_stats(states, actions, cam_names, n_steps, args.fps)
        row: Dict[str, Any] = {
            "episode_index": ep_idx,
            "data/chunk_index": 0,
            "data/file_index": ep_idx,
            "dataset_from_index": total_frames,
            "dataset_to_index": total_frames + n_steps,
            "tasks": [task],
            "length": n_steps,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for cam_name in cam_names:
            vk = f"observation.images.{cam_name}"
            row[f"videos/{vk}/chunk_index"] = 0
            row[f"videos/{vk}/file_index"] = ep_idx
            row[f"videos/{vk}/from_timestamp"] = 0.0
            row[f"videos/{vk}/to_timestamp"] = float((n_steps - 1) / args.fps)

        for feat_key, feat_stats in ep_stats.items():
            for stat_name, stat_val in feat_stats.items():
                row[f"stats/{feat_key}/{stat_name}"] = stat_val

        episodes_rows.append(row)
        total_frames += n_steps
        logger.info("  Episode {}: {} steps from {}", ep_idx, n_steps, ep_path.name)

    n_episodes = len(episodes_rows)
    if n_episodes == 0:
        logger.error("All episodes were empty")
        raise SystemExit(1)

    # --- Write meta/tasks.parquet ---
    tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task"))
    tasks_df.to_parquet(str(meta_dir / "tasks.parquet"))

    # --- Write meta/episodes/chunk-000/file-000.parquet ---
    episodes_df = pd.DataFrame(episodes_rows)
    episodes_table = pa.Table.from_pandas(episodes_df, preserve_index=False)
    pq.write_table(episodes_table, str(episodes_meta_dir / "file-000.parquet"))

    # --- Write meta/stats.json (aggregate stats) ---
    stats = compute_stats(all_states, all_actions)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- Write meta/info.json ---
    detected_codec = video_codec or "hevc"
    features: Dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": state_names,
            "fps": args.fps,
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": action_names,
            "fps": args.fps,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": args.fps},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": args.fps},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": args.fps},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": args.fps},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": args.fps},
    }
    for cam_name in cam_names:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": args.fps,
                "video.codec": detected_codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": args.robot_type,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "data_files_size_in_mb": round(total_data_bytes / 1e6, 1),
        "video_files_size_in_mb": round(total_video_bytes / 1e6, 1),
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info("=" * 50)
    logger.info("LeRobot v3.0 dataset written to: {}", output_dir)
    logger.info("  Episodes: {}, Total frames: {}", n_episodes, total_frames)
    logger.info("  State dim: {}, Action dim: {}", state_dim, action_dim)
    logger.info("  Cameras: {}", cam_names)
    logger.info("  Video codec: {}", detected_codec)

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
