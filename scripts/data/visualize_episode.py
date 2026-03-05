"""Visualize a recorded episode using Rerun.

Displays joint trajectories, gripper state, EE pose, and camera video
in a synchronized timeline viewer.

Usage:
    uv run scripts/data/visualize_episode.py --episode_dir recordings/episode_20260304_153045_0001
    uv run scripts/data/visualize_episode.py --episode_dir recordings/episode_20260304_153045_0001 --no-video
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger


@dataclass
class Args:
    episode_dir: str
    no_video: bool = False
    timeline_name: str = "step"


def main(args: Args) -> None:
    try:
        import rerun as rr
    except ImportError:
        logger.error("rerun-sdk not installed. Run: uv add rerun-sdk")
        raise SystemExit(1) from None

    episode_dir = Path(args.episode_dir)
    if not episode_dir.exists():
        logger.error("Episode directory not found: {}", episode_dir)
        raise SystemExit(1)

    # Load metadata
    metadata_path = episode_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        logger.info("Episode: {} steps, {:.1f}s", metadata.get("num_steps", "?"), metadata.get("duration_s", 0))
    else:
        metadata = {}

    # Load timestamps
    timestamps_path = episode_dir / "timestamps.npy"
    timestamps = np.load(str(timestamps_path)) if timestamps_path.exists() else None

    rr.init(f"limb — {episode_dir.name}", spawn=True)

    # Log arm states
    arm_names = metadata.get("arms", [])
    if not arm_names:
        # Auto-detect from files
        arm_names = [p.stem.replace("_states", "") for p in episode_dir.glob("*_states.npz")]

    for arm_name in arm_names:
        states_path = episode_dir / f"{arm_name}_states.npz"
        if not states_path.exists():
            continue

        states = dict(np.load(str(states_path)))
        n_steps = len(next(iter(states.values())))
        logger.info("Logging {}: {} steps, keys={}", arm_name, n_steps, list(states.keys()))

        for i in range(n_steps):
            rr.set_time_sequence(args.timeline_name, i)
            if timestamps is not None and i < len(timestamps):
                rr.set_time_seconds("wall_clock", timestamps[i])

            if "joint_pos" in states:
                for j, val in enumerate(states["joint_pos"][i]):
                    rr.log(f"{arm_name}/joint_pos/j{j}", rr.Scalar(val))

            if "joint_vel" in states:
                for j, val in enumerate(states["joint_vel"][i]):
                    rr.log(f"{arm_name}/joint_vel/j{j}", rr.Scalar(val))

            if "gripper_pos" in states:
                rr.log(f"{arm_name}/gripper_pos", rr.Scalar(states["gripper_pos"][i].item()))

            if "ee_pose" in states:
                ee = states["ee_pose"][i]
                if ee.shape[0] >= 7:
                    # wxyz + xyz
                    rr.log(
                        f"{arm_name}/ee_pose",
                        rr.Transform3D(
                            translation=ee[4:7],
                            rotation=rr.Quaternion(xyzw=[ee[1], ee[2], ee[3], ee[0]]),
                        ),
                    )

        # Log actions
        actions_path = episode_dir / f"{arm_name}_actions.npz"
        if actions_path.exists():
            actions = dict(np.load(str(actions_path)))
            for i in range(len(next(iter(actions.values())))):
                rr.set_time_sequence(args.timeline_name, i)
                if "pos" in actions:
                    for j, val in enumerate(actions["pos"][i]):
                        rr.log(f"{arm_name}/action/j{j}", rr.Scalar(val))

    # Log camera video frames
    if not args.no_video:
        cam_names = metadata.get("cameras", [])
        if not cam_names:
            cam_names = [p.stem for p in episode_dir.glob("*.mp4")]

        for cam_name in cam_names:
            video_path = episode_dir / f"{cam_name}.mp4"
            if not video_path.exists():
                continue

            cam_timestamps_path = episode_dir / f"{cam_name}_timestamps.npy"
            cam_timestamps = np.load(str(cam_timestamps_path)) if cam_timestamps_path.exists() else None

            import cv2

            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            logger.info("Logging video: {}", cam_name)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rr.set_time_sequence(args.timeline_name, frame_idx)
                if cam_timestamps is not None and frame_idx < len(cam_timestamps):
                    rr.set_time_seconds("wall_clock", cam_timestamps[frame_idx])

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log(f"cameras/{cam_name}", rr.Image(rgb))
                frame_idx += 1

            cap.release()
            logger.info("  {} frames logged", frame_idx)

    logger.info("Rerun viewer launched. Use timeline scrubber to navigate.")


if __name__ == "__main__":
    main(tyro.cli(Args))
