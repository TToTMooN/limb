"""Convert limb raw recordings to WebDataset tar format.

WebDataset stores training samples as consecutive files inside .tar archives,
designed for efficient streaming training (no random access overhead).
Each sample is a group of files sharing a key prefix:

    sample_000000.state.npy    # (state_dim,) float32
    sample_000000.action.npy   # (action_dim,) float32
    sample_000000.jpg          # RGB image from primary camera
    sample_000000.json         # metadata (timestamp, episode_idx, frame_idx)

Shards are split into configurable sizes (default 1000 samples per tar).
Stats (mean/std/min/max) are computed with Welford's online algorithm and
written alongside the shards.

Usage:
    uv run limb convert-webdataset --input-dir recordings/task --output-dir datasets/task_wds
    uv run limb convert-webdataset --input-dir recordings/task --output-dir datasets/task_wds --image-size 224

Output structure::

    datasets/task_wds/
        shard-000000.tar
        shard-000001.tar
        stats.json
        meta.json
"""

from __future__ import annotations

import io
import json
import tarfile
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
    find_episodes,
    load_episode,
)


@dataclass
class Args:
    """Convert raw limb recordings to WebDataset .tar shards."""

    input_dir: str
    output_dir: str
    task: Optional[str] = None
    samples_per_shard: int = 1000
    image_size: Optional[int] = None  # resize images to NxN (None = original)
    jpeg_quality: int = 90
    fps: int = 30
    success_only: bool = False
    include_depth: bool = False
    camera: Optional[str] = None  # primary camera name (None = first found)


class WelfordStats:
    """Welford's online algorithm for numerically stable mean/std."""

    def __init__(self, dim: int) -> None:
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min_val = np.full(dim, np.inf, dtype=np.float64)
        self.max_val = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        self.min_val = np.minimum(self.min_val, x)
        self.max_val = np.maximum(self.max_val, x)

    def finalize(self) -> Dict:
        std = np.sqrt(self.m2 / max(self.n, 1))
        return {
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "min": self.min_val.tolist(),
            "max": self.max_val.tolist(),
            "count": self.n,
        }


def _encode_jpeg(rgb: np.ndarray, quality: int = 90, size: Optional[int] = None) -> bytes:
    """Encode RGB numpy array to JPEG bytes, optionally resizing."""
    import cv2

    if size is not None:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _add_to_tar(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    """Add bytes to a tar archive under the given name."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _extract_frames(video_path: Path) -> List[np.ndarray]:
    """Extract all frames from a video file as RGB numpy arrays."""
    import cv2

    frames = []
    cap = cv2.VideoCapture(str(video_path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main(args: Args) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = find_episodes(input_dir, args.success_only)
    if not episodes:
        logger.error("No episodes found in {}", input_dir)
        raise SystemExit(1)

    logger.info("Found {} episodes in {}", len(episodes), input_dir)

    # Determine structure from first episode
    first_ep = load_episode(episodes[0])
    arm_names = sorted(first_ep["arms"].keys())
    cam_names = [c["name"] for c in first_ep["cameras"]]
    task = args.task or first_ep["metadata"].get("task_instruction", "")

    # Pick primary camera for image samples
    primary_cam = args.camera
    if primary_cam is None and cam_names:
        primary_cam = cam_names[0]
        logger.info("Using primary camera: {}", primary_cam)

    state_names = build_state_names(arm_names, first_ep)
    action_names = build_action_names(arm_names, first_ep)
    state_dim = len(state_names)
    action_dim = len(action_names)

    logger.info("State dim: {}, Action dim: {}, Cameras: {}", state_dim, action_dim, cam_names)

    # Online stats
    state_stats = WelfordStats(state_dim) if state_dim > 0 else None
    action_stats = WelfordStats(action_dim) if action_dim > 0 else None

    # Sharding state
    shard_idx = 0
    sample_in_shard = 0
    global_sample_idx = 0
    total_samples = 0
    tar: Optional[tarfile.TarFile] = None

    def _open_shard() -> tarfile.TarFile:
        nonlocal shard_idx
        path = output_dir / f"shard-{shard_idx:06d}.tar"
        return tarfile.open(str(path), "w")

    tar = _open_shard()

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

        timestamps = episode["timestamps"][:n_steps] if episode["timestamps"] is not None else None

        # Extract video frames for the primary camera
        frames: Optional[List[np.ndarray]] = None
        if primary_cam:
            video_path = ep_path / f"{primary_cam}.mp4"
            if video_path.exists():
                frames = _extract_frames(video_path)

        # Also extract frames for all other cameras
        other_cam_frames: Dict[str, List[np.ndarray]] = {}
        for cname in cam_names:
            if cname == primary_cam:
                continue
            vid = ep_path / f"{cname}.mp4"
            if vid.exists():
                other_cam_frames[cname] = _extract_frames(vid)

        for step in range(n_steps):
            key = f"sample_{global_sample_idx:08d}"

            # State and action as .npy
            state_buf = io.BytesIO()
            np.save(state_buf, states[step])
            _add_to_tar(tar, f"{key}.state.npy", state_buf.getvalue())

            action_buf = io.BytesIO()
            np.save(action_buf, actions[step])
            _add_to_tar(tar, f"{key}.action.npy", action_buf.getvalue())

            # Primary camera image as JPEG
            if frames is not None and step < len(frames):
                jpg = _encode_jpeg(frames[step], args.jpeg_quality, args.image_size)
                _add_to_tar(tar, f"{key}.jpg", jpg)

            # Additional camera images
            for cname, cframes in other_cam_frames.items():
                if step < len(cframes):
                    jpg = _encode_jpeg(cframes[step], args.jpeg_quality, args.image_size)
                    _add_to_tar(tar, f"{key}.{cname}.jpg", jpg)

            # Metadata JSON
            meta = {
                "episode_index": ep_idx,
                "frame_index": step,
                "task": task,
            }
            if timestamps is not None:
                meta["timestamp"] = float(timestamps[step])
            meta_bytes = json.dumps(meta).encode()
            _add_to_tar(tar, f"{key}.json", meta_bytes)

            # Update stats
            if state_stats:
                state_stats.update(states[step].astype(np.float64))
            if action_stats:
                action_stats.update(actions[step].astype(np.float64))

            global_sample_idx += 1
            sample_in_shard += 1
            total_samples += 1

            # Rotate shard
            if sample_in_shard >= args.samples_per_shard:
                tar.close()
                shard_idx += 1
                sample_in_shard = 0
                tar = _open_shard()

        logger.info("  Episode {}: {} samples", ep_idx, n_steps)

    if tar is not None:
        tar.close()
    # Remove empty last shard
    if sample_in_shard == 0 and shard_idx > 0:
        empty_shard = output_dir / f"shard-{shard_idx:06d}.tar"
        if empty_shard.exists():
            empty_shard.unlink()
            shard_idx -= 1

    # Write stats
    stats = {}
    if state_stats:
        stats["observation.state"] = state_stats.finalize()
    if action_stats:
        stats["action"] = action_stats.finalize()

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Write meta
    meta_info = {
        "format": "webdataset",
        "total_samples": total_samples,
        "total_shards": shard_idx + 1,
        "samples_per_shard": args.samples_per_shard,
        "total_episodes": len(episodes),
        "task": task,
        "fps": args.fps,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_names": state_names,
        "action_names": action_names,
        "cameras": cam_names,
        "primary_camera": primary_cam,
        "image_size": args.image_size,
        "jpeg_quality": args.jpeg_quality,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    logger.info("=" * 50)
    logger.info("WebDataset written to: {}", output_dir)
    logger.info("  Shards: {}, Samples: {}", shard_idx + 1, total_samples)
    logger.info("  State dim: {}, Action dim: {}", state_dim, action_dim)


if __name__ == "__main__":
    main(tyro.cli(Args))
