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

import concurrent.futures
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
from limb.data.resample import (
    detect_source_fps,
    resample_state_action,
    resample_video,
)

CODEBASE_VERSION = "v3.0"


@dataclass
class Args:
    input_dir: str
    output_dir: str
    task: Optional[str] = None
    robot_type: str = "yam"
    # Output dataset rate. None → use each episode's detected real rate (no
    # resampling, just honest labeling). Set explicitly to resample state /
    # action / video onto a regular target-fps grid.
    target_fps: Optional[int] = None
    # Action dims that use nearest-neighbor instead of linear interp during
    # resampling. Typically the gripper dims (binarized 0 ↔ max). Default
    # matches the YAM bimanual action layout (left_gripper=6, right_gripper=13).
    nearest_action_dims: tuple = (6, 13)
    success_only: bool = False
    push_to_hub: Optional[str] = None
    # Episodes are independent — the expensive per-episode work (resample +
    # 3 video re-encodes) parallelizes cleanly. Default 2 to stay well within
    # consumer-GPU NVENC concurrency limits while still ~2x'ing throughput.
    # Set to 1 for fully-deterministic-order logging.
    max_workers: int = 2
    # Deprecated: pre-resample, the legacy --fps flag relabeled to a hardcoded
    # rate without touching the data. Kept for backwards compatibility with
    # existing scripts; if --target-fps is set, it takes precedence.
    fps: Optional[int] = None


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
    fps: float,
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

    # Resolve fps policy. `--target-fps` (new) takes precedence; `--fps` is the
    # legacy alias kept for backwards compatibility. When both are None we
    # auto-detect each episode's real source fps from its timestamps.npy.
    legacy_fps = args.fps
    target_fps_override: Optional[int] = args.target_fps if args.target_fps is not None else legacy_fps
    if legacy_fps is not None and args.target_fps is None:
        logger.warning(
            "--fps {} is deprecated; use --target-fps to make the resampling intent explicit. "
            "Treating as --target-fps {}.",
            legacy_fps, legacy_fps,
        )
    elif target_fps_override is None:
        logger.info("No --target-fps: auto-detecting each episode's source fps; no resampling.")

    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episodes_rows: List[Dict[str, Any]] = []
    total_frames = 0
    video_codec: Optional[str] = None
    total_data_bytes = 0
    total_video_bytes = 0
    per_episode_output_fps: List[float] = []
    zoh_dims = tuple(int(d) for d in args.nearest_action_dims)

    # ---- Phase 1: heavy work (load + resample + video re-encode) ----
    # Run in a thread pool so each worker holds its own ffmpeg subprocess.
    # The work is I/O and ffmpeg-bound — both release the GIL — so threads
    # are enough; we don't need ProcessPool. Set max_workers low enough that
    # NVENC concurrency stays comfortable (consumer GPUs typically handle
    # 2-3 simultaneous H.265 encode sessions cleanly).
    def _heavy_episode(ep_idx: int, ep_path: Path) -> Optional[Dict[str, Any]]:
        episode = load_episode(ep_path)
        n_steps = len(episode["timestamps"]) if episode["timestamps"] is not None else 0
        if n_steps == 0:
            logger.warning("Skipping empty episode: {}", ep_path.name)
            return None

        states = build_state_vector(episode, arm_names)
        actions = build_action_vector(episode, arm_names)
        n_steps = min(len(states), len(actions)) if len(actions) > 0 else len(states)
        states = states[:n_steps]
        actions = actions[:n_steps] if len(actions) > 0 else np.zeros((n_steps, action_dim), dtype=np.float32)
        timestamps = np.asarray(episode["timestamps"], dtype=np.float64)[:n_steps]

        source_fps = detect_source_fps(timestamps)
        if source_fps <= 0:
            logger.warning("Skipping {}: invalid timestamps (duration <= 0)", ep_path.name)
            return None

        do_resample = (
            target_fps_override is not None
            and abs(target_fps_override - source_fps) / source_fps > 0.01
        )
        output_fps = float(target_fps_override) if target_fps_override is not None else source_fps

        # Optional DAgger phase metadata (None when episode lacks phase.npy).
        # These align 1:1 with the source state/action grid by construction.
        src_phase = episode.get("phase")
        src_corr_idx = episode.get("correction_index")

        if do_resample:
            logger.info(
                "  Episode {}: resample {} steps @ {:.2f} Hz -> @ {:.2f} Hz",
                ep_idx, n_steps, source_fps, output_fps,
            )
            tgt_rel, states, actions = resample_state_action(
                timestamps, states, actions, output_fps, zoh_action_dims=zoh_dims,
            )
            n_steps_out = len(tgt_rel)
            # ZOH-resample phase metadata onto the target grid. Phase labels
            # are discrete; ZOH is the only sensible choice and matches what
            # we do for the gripper and video streams.
            if src_phase is not None or src_corr_idx is not None:
                from limb.data.resample import _zoh_indices
                src_rel = timestamps - timestamps[0]
                zoh = _zoh_indices(src_rel, tgt_rel)
                if src_phase is not None and len(src_phase) >= len(timestamps):
                    src_phase = src_phase[: len(timestamps)][zoh]
                if src_corr_idx is not None and len(src_corr_idx) >= len(timestamps):
                    src_corr_idx = src_corr_idx[: len(timestamps)][zoh]
        else:
            tgt_rel = None
            logger.info(
                "  Episode {}: {} steps @ {:.2f} Hz (no resample{})",
                ep_idx, n_steps, output_fps,
                ", labels-only update" if target_fps_override is None else "",
            )
            n_steps_out = n_steps
            # Truncate to n_steps in case state/action were clipped earlier.
            if src_phase is not None and len(src_phase) > n_steps_out:
                src_phase = src_phase[:n_steps_out]
            if src_corr_idx is not None and len(src_corr_idx) > n_steps_out:
                src_corr_idx = src_corr_idx[:n_steps_out]

        # Videos: copy verbatim when no resampling, re-encode otherwise.
        # Cameras are processed sequentially inside the worker so each
        # worker only ever has 1 ffmpeg subprocess running at a time.
        worker_video_size = 0
        worker_codec: Optional[str] = None
        for cam in episode["cameras"]:
            src = cam["video_path"]
            dst = (
                output_dir / "videos" / f"observation.images.{cam['name']}"
                / "chunk-000" / f"file-{ep_idx:03d}.mp4"
            )
            if do_resample:
                resample_video(src, dst, timestamps, tgt_rel, output_fps)
            else:
                shutil.copy2(str(src), str(dst))
            worker_video_size += dst.stat().st_size
            if worker_codec is None:
                worker_codec = _probe_video_codec(dst)

        return {
            "ep_idx": ep_idx,
            "states": states,
            "actions": actions,
            "n_steps_out": n_steps_out,
            "output_fps": output_fps,
            "video_size": worker_video_size,
            "codec": worker_codec,
            "phase": src_phase,                  # may be None
            "correction_index": src_corr_idx,    # may be None
        }

    results_by_idx: Dict[int, Dict[str, Any]] = {}
    if args.max_workers <= 1:
        for ep_idx, ep_path in enumerate(episodes):
            res = _heavy_episode(ep_idx, ep_path)
            if res is not None:
                results_by_idx[ep_idx] = res
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_to_idx = {
                pool.submit(_heavy_episode, ep_idx, ep_path): ep_idx
                for ep_idx, ep_path in enumerate(episodes)
            }
            for fut in concurrent.futures.as_completed(future_to_idx):
                ep_idx = future_to_idx[fut]
                res = fut.result()
                if res is not None:
                    results_by_idx[ep_idx] = res

    # ---- Phase 2: serialized parquet writes + metadata aggregation ----
    # Strict ascending ep_idx order so `index` and `dataset_from_index` are
    # contiguous and reproducible. Parquet writes are cheap (~ms each).
    for ep_idx in sorted(results_by_idx.keys()):
        res = results_by_idx[ep_idx]
        states = res["states"]
        actions = res["actions"]
        n_steps_out = res["n_steps_out"]
        output_fps = res["output_fps"]

        per_episode_output_fps.append(output_fps)
        all_states.append(states)
        all_actions.append(actions)

        table_data = {
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
            "episode_index": pa.array(np.full(n_steps_out, ep_idx, dtype=np.int64)),
            "frame_index": pa.array(np.arange(n_steps_out, dtype=np.int64)),
            "timestamp": pa.array((np.arange(n_steps_out, dtype=np.float64) / output_fps).astype(np.float32)),
            "index": pa.array(np.arange(total_frames, total_frames + n_steps_out, dtype=np.int64)),
            "task_index": pa.array(np.zeros(n_steps_out, dtype=np.int64)),
        }
        # DAgger phase columns are appended only when the source episode had
        # them, so non-DAgger datasets stay schema-clean.
        if res.get("phase") is not None:
            phase_arr = np.asarray(res["phase"])
            if len(phase_arr) >= n_steps_out:
                table_data["phase"] = pa.array(phase_arr[:n_steps_out].astype(str).tolist())
        if res.get("correction_index") is not None:
            ci = np.asarray(res["correction_index"])
            if len(ci) >= n_steps_out:
                table_data["correction_index"] = pa.array(ci[:n_steps_out].astype(np.int32))
        table = pa.table(table_data)
        parquet_path = data_dir / f"file-{ep_idx:03d}.parquet"
        pq.write_table(table, str(parquet_path), compression="snappy")
        total_data_bytes += parquet_path.stat().st_size
        total_video_bytes += res["video_size"]
        if video_codec is None:
            video_codec = res["codec"]

        ep_stats = _compute_episode_stats(states, actions, cam_names, n_steps_out, output_fps)
        row: Dict[str, Any] = {
            "episode_index": ep_idx,
            "data/chunk_index": 0,
            "data/file_index": ep_idx,
            "dataset_from_index": total_frames,
            "dataset_to_index": total_frames + n_steps_out,
            "tasks": [task],
            "length": n_steps_out,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for cam_name in cam_names:
            vk = f"observation.images.{cam_name}"
            row[f"videos/{vk}/chunk_index"] = 0
            row[f"videos/{vk}/file_index"] = ep_idx
            row[f"videos/{vk}/from_timestamp"] = 0.0
            row[f"videos/{vk}/to_timestamp"] = float((n_steps_out - 1) / output_fps)
        for feat_key, feat_stats in ep_stats.items():
            for stat_name, stat_val in feat_stats.items():
                row[f"stats/{feat_key}/{stat_name}"] = stat_val
        episodes_rows.append(row)
        total_frames += n_steps_out

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
    # info.json's top-level `fps` is a single number, but our per-episode fps
    # may differ when --target-fps is unset and the recordings drift. Pick the
    # mode (most common) and warn if any episode disagrees by > 1%.
    uniq_fps, counts = np.unique(np.round(per_episode_output_fps, 3), return_counts=True)
    dataset_fps = float(uniq_fps[int(np.argmax(counts))])
    if len(uniq_fps) > 1:
        spread = (uniq_fps.max() - uniq_fps.min()) / dataset_fps
        if spread > 0.01:
            logger.warning(
                "Episodes have inconsistent output fps (range {:.2f}..{:.2f} Hz). "
                "Writing info.json:fps = {:.2f} (modal) — consumer code that joins "
                "across episodes may need to use the per-episode timestamp column.",
                uniq_fps.min(), uniq_fps.max(), dataset_fps,
            )
    info_fps = round(dataset_fps) if abs(dataset_fps - round(dataset_fps)) < 1e-3 else dataset_fps

    detected_codec = video_codec or "hevc"
    features: Dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": state_names,
            "fps": info_fps,
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": action_names,
            "fps": info_fps,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": info_fps},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": info_fps},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": info_fps},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": info_fps},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": info_fps},
    }
    for cam_name in cam_names:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": info_fps,
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
        "fps": info_fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "data_files_size_in_mb": round(total_data_bytes / 1e6, 1),
        "video_files_size_in_mb": round(total_video_bytes / 1e6, 1),
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # --- Write README.md dataset card ---
    _write_dataset_card(
        output_dir,
        task=task,
        robot_type=args.robot_type,
        n_episodes=n_episodes,
        total_frames=total_frames,
        fps=info_fps,
        cam_names=cam_names,
        state_names=state_names,
        action_names=action_names,
        detected_codec=detected_codec,
    )

    logger.info("=" * 50)
    logger.info("LeRobot v3.0 dataset written to: {}", output_dir)
    logger.info("  Episodes: {}, Total frames: {}", n_episodes, total_frames)
    logger.info("  State dim: {}, Action dim: {}", state_dim, action_dim)
    logger.info("  Cameras: {}", cam_names)
    logger.info("  Video codec: {}", detected_codec)

    if args.push_to_hub:
        _push_to_hub(output_dir, args.push_to_hub)


def _write_dataset_card(
    output_dir: Path,
    *,
    task: str,
    robot_type: str,
    n_episodes: int,
    total_frames: int,
    fps: float,
    cam_names: List[str],
    state_names: List[str],
    action_names: List[str],
    detected_codec: str,
) -> None:
    """Generate a HuggingFace dataset card (README.md) for the converted dataset."""
    state_dim = len(state_names)
    action_dim = len(action_names)

    state_table = "\n".join(
        f"| {i} | `{name}` |" for i, name in enumerate(state_names)
    )
    action_table = "\n".join(
        f"| {i} | `{name}` |" for i, name in enumerate(action_names)
    )
    cam_list = "\n".join(f"| `{c}` |" for c in cam_names)

    card = f"""---
license: apache-2.0
task_categories:
  - robotics
tags:
  - LeRobot
  - {robot_type}
  - teleop
  - manipulation
configs:
  - config_name: default
    data_files: data/**/*.parquet
---

# {output_dir.name}

Teleoperation dataset: **{task}**

Collected with [limb](https://github.com/TToTMooN/limb) on {robot_type} arms.

## Dataset summary

| Field | Value |
|-------|-------|
| Robot | {robot_type} |
| Episodes | {n_episodes} |
| Total frames | {total_frames:,} |
| FPS | {fps} Hz |
| Task | {task} |
| Format | LeRobot v3.0 |

## Cameras

| Name |
|------|
{cam_list}

Video codec: {detected_codec}.

## State space (`observation.state`, shape `[{state_dim}]`)

| Index | Name |
|-------|------|
{state_table}

## Action space (`action`, shape `[{action_dim}]`)

| Index | Name |
|-------|------|
{action_table}

State and action are 1:1 index-aligned.

> **Note:** Gripper dimensions may use different units between state (raw motor
> radians) and action (teleoperator command space). The robot's JointMapper
> rescales between these at command time. Left and right grippers may also have
> different value ranges if the physical hardware travel differs.

## Usage

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("<repo_id>")
print(ds.num_episodes, ds.num_frames, ds[0]["observation.state"].shape)
```

## License

Apache 2.0
"""
    (output_dir / "README.md").write_text(card)
    logger.info("Dataset card written: {}", output_dir / "README.md")


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
