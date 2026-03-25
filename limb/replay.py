"""Motion replay — replay recorded episodes on hardware for verification.

Inspired by Raiden's `rd replay` command. Streams joint commands from a
recorded episode to the physical robot at configurable speed. Useful for
checking recording quality before conversion.

Usage:
    uv run limb replay --episode-dir recordings/session/episode_20260304_153045_0001
    uv run limb replay --episode-dir recordings/session/episode_... --speed 0.5
"""

from __future__ import annotations

import json
import signal
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from loguru import logger

_shutdown_requested = False


def _sigint_handler(signum: int, frame: Any) -> None:
    global _shutdown_requested
    if _shutdown_requested:
        raise KeyboardInterrupt
    _shutdown_requested = True


def _load_episode_data(episode_dir: Path) -> Dict[str, Any]:
    """Load arm states and timestamps from a recorded episode."""
    data: Dict[str, Any] = {}

    # Load metadata
    meta_path = episode_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)
    else:
        raise FileNotFoundError(f"No metadata.json in {episode_dir}")

    # Load timestamps
    ts_path = episode_dir / "timestamps.npy"
    if ts_path.exists():
        data["timestamps"] = np.load(str(ts_path))
    else:
        raise FileNotFoundError(f"No timestamps.npy in {episode_dir}")

    # Load arm states
    data["arms"] = {}
    for states_path in sorted(episode_dir.glob("*_states.npz")):
        arm_name = states_path.stem.replace("_states", "")
        arm_data = dict(np.load(str(states_path)))
        data["arms"][arm_name] = arm_data

    # Load actions (preferred for replay since they include gripper)
    data["actions"] = {}
    for actions_path in sorted(episode_dir.glob("*_actions.npz")):
        arm_name = actions_path.stem.replace("_actions", "")
        data["actions"][arm_name] = dict(np.load(str(actions_path)))

    return data


def replay_episode(
    episode_dir: str,
    config_path: List[str],
    speed: float = 1.0,
    log_level: str = "INFO",
) -> None:
    """Replay a recorded episode on hardware.

    Parameters
    ----------
    episode_dir : str
        Path to the episode directory containing states/actions.
    config_path : list[str]
        Robot config YAML paths (needed to initialize hardware).
    speed : float
        Playback speed multiplier (0.5 = half speed, 2.0 = double).
    log_level : str
        Logging level.
    """
    global _shutdown_requested
    _shutdown_requested = False

    from limb.envs.configs.instantiate import instantiate
    from limb.envs.configs.loader import DictLoader
    from limb.robots.robot import Robot
    from limb.robots.utils import Rate
    from limb.utils.launch_utils import (
        cleanup_processes,
        initialize_robots,
        setup_can_interfaces,
        setup_logging,
    )

    setup_logging(level=log_level)

    ep_path = Path(episode_dir)
    if not ep_path.exists():
        logger.error("Episode directory not found: {}", episode_dir)
        return

    logger.info("Loading episode data from: {}", ep_path)
    episode_data = _load_episode_data(ep_path)

    timestamps = episode_data["timestamps"]
    n_steps = len(timestamps)
    if n_steps == 0:
        logger.error("Episode has no timesteps (empty timestamps.npy)")
        return
    arm_names = sorted(episode_data["arms"].keys())
    if not arm_names:
        logger.error("Episode has no arm data")
        return
    duration = timestamps[-1] - timestamps[0]

    logger.info(
        "Episode: {} steps, {:.1f}s duration, arms: {}, speed: {:.1f}x",
        n_steps,
        duration,
        arm_names,
        speed,
    )

    # Use actions if available (includes gripper), fall back to states
    use_actions = bool(episode_data["actions"])
    if use_actions:
        logger.info("Replaying from recorded actions (joint_pos + gripper)")
    else:
        logger.info("Replaying from recorded states (joint_pos only)")

    # Load config and initialize hardware
    configs_dict = DictLoader.load(config_path)
    configs_dict.pop("agent", None)
    configs_dict.pop("sensors", None)
    configs_dict.pop("api_servers", None)
    configs_dict.pop("collection", None)
    configs_dict.pop("recording", None)
    main_config = instantiate(configs_dict)

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    server_processes: list = []
    robots: Dict[str, Robot] = {}

    try:
        setup_can_interfaces()
        robots = initialize_robots(main_config.robots, server_processes)

        # Move to first pose over 3 seconds
        first_targets = {}
        for arm_name in arm_names:
            if arm_name not in robots:
                logger.warning("Arm '{}' in recording but not in robot config, skipping", arm_name)
                continue
            if use_actions and arm_name in episode_data["actions"] and "pos" in episode_data["actions"][arm_name]:
                first_targets[arm_name] = episode_data["actions"][arm_name]["pos"][0]
            elif "joint_pos" in episode_data["arms"][arm_name]:
                jp = episode_data["arms"][arm_name]["joint_pos"][0]
                gp = episode_data["arms"].get(arm_name, {}).get("gripper_pos", None)
                if gp is not None and len(gp) > 0:
                    first_targets[arm_name] = np.concatenate([jp, gp[0]])
                else:
                    first_targets[arm_name] = jp

        if first_targets:
            logger.info("Moving to first pose over 3.0s...")

            def _move_one(name: str, robot: Robot, target: np.ndarray) -> None:
                try:
                    robot.move_joints(target, 3.0)
                except Exception as e:
                    logger.warning("Could not move '{}' to start pose: {}", name, e)

            threads = []
            for name, robot in robots.items():
                if name in first_targets:
                    t = threading.Thread(target=_move_one, args=(name, robot, first_targets[name]), daemon=True)
                    t.start()
                    threads.append(t)
            for t in threads:
                t.join(timeout=5.0)

        # Stream recorded commands
        logger.info("Starting replay...")
        effective_hz = (n_steps / duration) * speed if duration > 0 else 30.0
        rate = Rate(effective_hz, rate_name="replay")

        for step_idx in range(n_steps):
            if _shutdown_requested:
                logger.info("Shutdown requested, stopping replay.")
                break

            for arm_name in arm_names:
                if arm_name not in robots:
                    continue

                if use_actions and arm_name in episode_data["actions"] and "pos" in episode_data["actions"][arm_name]:
                    target = episode_data["actions"][arm_name]["pos"][step_idx]
                else:
                    jp = episode_data["arms"][arm_name]["joint_pos"][step_idx]
                    gp = episode_data["arms"][arm_name].get("gripper_pos", None)
                    if gp is not None and step_idx < len(gp):
                        target = np.concatenate([jp, gp[step_idx]])
                    else:
                        target = jp

                robots[arm_name].command_joint_pos(target)

            rate.sleep()

            # Progress update every second
            if step_idx % max(1, int(effective_hz)) == 0:
                progress = (step_idx + 1) / n_steps * 100
                logger.info("Replay progress: {:.0f}% ({}/{})", progress, step_idx + 1, n_steps)

        logger.info("Replay complete.")

    except KeyboardInterrupt:
        logger.info("Replay interrupted.")
    except Exception as e:
        logger.error("Replay error: {}", e)
        raise
    finally:
        # Return to safe position
        for _name, robot in robots.items():
            try:
                robot.soft_release(2.0)
            except Exception:
                try:
                    robot.zero_torque_mode()
                except Exception:
                    pass

        cleanup_processes(None, server_processes)
        signal.signal(signal.SIGINT, original_sigint)
