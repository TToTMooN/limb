"""Non-destructive left/right sanity check for cameras and arms.

Spins up the same cameras + robots that the launch config would, then:

1. Captures one frame from each wrist camera, saves them to /tmp with the
   limb config-name in the filename, prints the serial. Open them and confirm
   each shows the gripper you expect.
2. Continuously reads joint state from both arms and prints the value of any
   joint that moves between polls. Manually nudge the LEFT physical arm and
   confirm only the line labeled ``left`` reports a delta (and vice versa).

The script never commands the robot — only reads. Safe to run with the arms
powered.

Usage::

    uv run scripts/diagnostics/check_left_right.py
    uv run scripts/diagnostics/check_left_right.py --config-path configs/yam_xvla_bimanual.yaml
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import tyro
from loguru import logger

from limb.envs.configs.instantiate import instantiate
from limb.envs.configs.loader import DictLoader
from limb.utils.launch_utils import (
    cleanup_processes,
    initialize_robots,
    initialize_sensors,
    setup_can_interfaces,
    setup_logging,
)

_shutdown_requested = False


def _sigint_handler(signum: int, frame: Any) -> None:
    global _shutdown_requested
    _shutdown_requested = True


@dataclass
class Args:
    config_path: Tuple[str, ...] = ("configs/yam_policy_bimanual.yaml",)
    poll_hz: float = 5.0
    duration_s: float = 60.0
    delta_threshold_rad: float = 0.005  # what counts as "moved"
    snapshot_dir: str = "/tmp/limb_lr_check"
    log_level: str = "INFO"


def main(args: Args) -> None:
    setup_logging(level=args.log_level)
    logger.info("=" * 70)
    logger.info("LEFT/RIGHT SANITY CHECK — robot is NOT commanded")
    logger.info("=" * 70)

    signal.signal(signal.SIGINT, _sigint_handler)
    server_processes: list = []

    try:
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])
        configs_dict.pop("agent", None)
        configs_dict.pop("api_servers", None)
        configs_dict.pop("collection", None)
        configs_dict.pop("recording", None)
        sensors_cfg = configs_dict.pop("sensors", None)
        main_config = instantiate(configs_dict)

        # ----- Step (B): Cameras ----- #
        logger.info("")
        logger.info("Initializing cameras...")
        camera_dict, _ = initialize_sensors(sensors_cfg, server_processes)

        snap_dir = Path(args.snapshot_dir)
        snap_dir.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info("─── Camera mapping (verify by opening snapshots) ───")
        for cam_name, client in camera_dict.items():
            data = client.read()
            rgb = np.asarray(data["images"]["rgb"])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out_path = snap_dir / f"{cam_name}.png"
            cv2.imwrite(str(out_path), bgr)
            serial = (data.get("metadata") or {}).get("serial_number", "?")
            logger.info("  {:<20s}  serial={}  snapshot={}", cam_name, serial, out_path)

        logger.info("")
        logger.info(
            "→ Open the PNGs and confirm: 'left_wrist_camera.png' shows the LEFT arm's gripper, "
            "'right_wrist_camera.png' shows the RIGHT arm's gripper. The head camera should look "
            "like the scene from the head/chest viewpoint."
        )

        # ----- Step (A): Arms ----- #
        setup_can_interfaces()
        logger.info("")
        logger.info("Initializing robots (read-only)...")
        robots = initialize_robots(main_config.robots, server_processes)

        # Baseline state
        baseline = {}
        for name, robot in robots.items():
            obs = robot.get_observations()
            baseline[name] = np.asarray(obs["joint_pos"])

        logger.info("")
        logger.info("─── Arm mapping (push one arm by hand at a time) ───")
        for name, jp in baseline.items():
            logger.info("  {:<10s}  joint_pos={}", name, np.round(jp, 3).tolist())

        logger.info("")
        logger.info(
            "→ Gently nudge the LEFT physical arm. The line below labeled 'left' should "
            "show non-zero deltas. If 'right' lights up when you push the left arm, the "
            "CAN-bus → arm mapping is inverted (swap can_follow_l ↔ can_follow_r in the "
            "robot YAMLs, or re-plug the USB-CAN adapters)."
        )
        logger.info("")
        logger.info("Polling for {:.0f}s at {:.1f} Hz — press Ctrl-C to stop early.", args.duration_s, args.poll_hz)

        dt = 1.0 / args.poll_hz
        deadline = time.time() + args.duration_s
        last_print = 0.0
        while time.time() < deadline and not _shutdown_requested:
            t = time.time()
            now = {}
            deltas = {}
            for name, robot in robots.items():
                obs = robot.get_observations()
                now[name] = np.asarray(obs["joint_pos"])
                deltas[name] = now[name] - baseline[name]
            # Only print when something has actually moved (avoid log spam)
            moved = {n: d for n, d in deltas.items() if np.abs(d).max() > args.delta_threshold_rad}
            if moved and (t - last_print) > 0.1:
                parts = []
                for name in sorted(robots.keys()):
                    d = deltas[name]
                    max_abs = np.abs(d).max()
                    marker = "★" if name in moved else " "
                    parts.append(f"  {marker} {name:<5s} max|Δ|={max_abs:.3f}  Δ={np.round(d, 3).tolist()}")
                logger.info("\n".join(parts))
                last_print = t
            time.sleep(dt)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cleanup_processes(None, server_processes)


if __name__ == "__main__":
    main(tyro.cli(Args))
