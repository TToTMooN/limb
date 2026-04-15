"""Visualize a recorded episode using Rerun.

Displays joint trajectories, gripper state, EE pose, and camera video
in a synchronized timeline viewer.

Usage:
    uv run limb visualize --episode-dir recordings/episode_20260304_153045_0001
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
        import rerun.blueprint as rrb
    except ImportError:
        logger.error("rerun-sdk not installed. Run: uv add rerun-sdk")
        raise SystemExit(1) from None

    episode_dir = Path(args.episode_dir)
    if not episode_dir.exists():
        logger.error("Episode directory not found: {}", episode_dir)
        raise SystemExit(1)

    metadata_path = episode_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        logger.info("Episode: {} steps, {:.1f}s", metadata.get("num_steps", "?"), metadata.get("duration_s", 0))
    else:
        metadata = {}

    timestamps_path = episode_dir / "timestamps.npy"
    timestamps = np.load(str(timestamps_path)) if timestamps_path.exists() else None

    arm_names = metadata.get("arms", [])
    if not arm_names:
        arm_names = sorted(p.stem.replace("_states", "") for p in episode_dir.glob("*_states.npz"))

    all_arms = {}
    for arm_name in arm_names:
        states_path = episode_dir / f"{arm_name}_states.npz"
        if not states_path.exists():
            continue
        arm = {"states": dict(np.load(str(states_path)))}
        actions_path = episode_dir / f"{arm_name}_actions.npz"
        if actions_path.exists():
            arm["actions"] = dict(np.load(str(actions_path)))
        all_arms[arm_name] = arm
        logger.info(
            "Logging {}: {} steps, keys={}",
            arm_name,
            len(next(iter(arm["states"].values()))),
            list(arm["states"].keys()),
        )

    views = []
    for arm_name in arm_names:
        if arm_name not in all_arms:
            continue
        states = all_arms[arm_name]["states"]
        if "joint_pos" in states:
            views.append(rrb.TimeSeriesView(name=f"{arm_name} joint_pos", origin=f"{arm_name}/joint_pos"))
        if "joint_vel" in states:
            views.append(rrb.TimeSeriesView(name=f"{arm_name} joint_vel", origin=f"{arm_name}/joint_vel"))
        if "gripper_pos" in states:
            views.append(rrb.TimeSeriesView(name=f"{arm_name} gripper", origin=f"{arm_name}/gripper"))
        if "actions" in all_arms[arm_name] and "pos" in all_arms[arm_name]["actions"]:
            views.append(rrb.TimeSeriesView(name=f"{arm_name} action", origin=f"{arm_name}/action"))

    cam_names = []
    if not args.no_video:
        cam_names = metadata.get("cameras", [])
        if not cam_names:
            cam_names = [p.stem for p in episode_dir.glob("*.mp4")]
        for cam_name in cam_names:
            views.append(rrb.Spatial2DView(name=cam_name, origin=f"cameras/{cam_name}"))

    blueprint = rrb.Blueprint(rrb.Grid(*views))
    rr.init(f"limb — {episode_dir.name}", spawn=True, default_blueprint=blueprint)
    rr.send_blueprint(blueprint)

    for arm_name, arm in all_arms.items():
        states = arm["states"]
        if "joint_pos" in states:
            for j in range(states["joint_pos"].shape[1]):
                rr.log(f"{arm_name}/joint_pos/j{j}", rr.SeriesLines(names=f"j{j}"), static=True)
        if "joint_vel" in states:
            for j in range(states["joint_vel"].shape[1]):
                rr.log(f"{arm_name}/joint_vel/j{j}", rr.SeriesLines(names=f"j{j}"), static=True)
        if "gripper_pos" in states:
            rr.log(f"{arm_name}/gripper/pos", rr.SeriesLines(names="gripper"), static=True)
        if "actions" in arm and "pos" in arm["actions"]:
            for j in range(arm["actions"]["pos"].shape[1]):
                rr.log(f"{arm_name}/action/j{j}", rr.SeriesLines(names=f"j{j}"), static=True)

    n_steps = (
        len(timestamps)
        if timestamps is not None
        else max(len(next(iter(arm["states"].values()))) for arm in all_arms.values())
    )
    for i in range(n_steps):
        rr.set_time(args.timeline_name, sequence=i)
        if timestamps is not None and i < len(timestamps):
            rr.set_time("wall_clock", timestamp=timestamps[i])

        for arm_name, arm in all_arms.items():
            states = arm["states"]
            if "joint_pos" in states and i < len(states["joint_pos"]):
                for j, val in enumerate(states["joint_pos"][i]):
                    rr.log(f"{arm_name}/joint_pos/j{j}", rr.Scalars(val))
            if "joint_vel" in states and i < len(states["joint_vel"]):
                for j, val in enumerate(states["joint_vel"][i]):
                    rr.log(f"{arm_name}/joint_vel/j{j}", rr.Scalars(val))
            if "gripper_pos" in states and i < len(states["gripper_pos"]):
                rr.log(f"{arm_name}/gripper/pos", rr.Scalars(states["gripper_pos"][i]))
            if "actions" in arm and "pos" in arm["actions"] and i < len(arm["actions"]["pos"]):
                for j, val in enumerate(arm["actions"]["pos"][i]):
                    rr.log(f"{arm_name}/action/j{j}", rr.Scalars(val))

    if not args.no_video:
        for cam_name in cam_names:
            video_path = episode_dir / f"{cam_name}.mp4"
            if not video_path.exists():
                continue

            cam_timestamps_path = episode_dir / f"{cam_name}_timestamps.npy"
            cam_timestamps = np.load(str(cam_timestamps_path)) if cam_timestamps_path.exists() else None
            if cam_timestamps is not None and len(cam_timestamps) > 0 and cam_timestamps[0] > 1e12:
                cam_timestamps = cam_timestamps / 1e3

            import cv2

            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            logger.info("Logging video: {}", cam_name)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rr.set_time(args.timeline_name, sequence=frame_idx)
                if cam_timestamps is not None and frame_idx < len(cam_timestamps):
                    rr.set_time("wall_clock", timestamp=cam_timestamps[frame_idx])

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log(f"cameras/{cam_name}", rr.Image(rgb))
                frame_idx += 1

            cap.release()
            logger.info("  {} frames logged", frame_idx)

    logger.info("Done logging. Viewer should be open. Press Ctrl+C to exit.")

    try:
        import signal

        signal.pause()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(tyro.cli(Args))
