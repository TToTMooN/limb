"""Unified CLI entry point for limb.

Inspired by Raiden's `rd` command, this provides a single `limb` command
with subcommands for all common operations:

    limb teleop    — Launch teleoperation
    limb record    — Launch data collection session
    limb devices   — Discover connected hardware
    limb replay    — Replay a recorded episode on hardware
    limb convert   — Convert raw recordings to LeRobot format
    limb visualize — Visualize a recorded episode with Rerun

Usage:
    uv run limb <command> [options]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tyro

# ── Subcommands ──────────────────────────────────────────────────────────


@dataclass
class TeleopCommand:
    """Launch teleoperation (no recording)."""

    config_path: Tuple[str, ...] = ("configs/yam_viser_bimanual.yaml",)
    log_level: str = "INFO"

    def run(self) -> None:
        from limb.envs.launch import Args, main

        main(Args(config_path=self.config_path, log_level=self.log_level))


@dataclass
class RecordCommand:
    """Launch a data collection session (teleop + recording)."""

    config_path: Tuple[str, ...] = ("configs/yam_gello_bimanual.yaml", "configs/collection.yaml")
    log_level: str = "INFO"

    def run(self) -> None:
        from limb.envs.launch import Args, main

        main(Args(config_path=self.config_path, log_level=self.log_level))


@dataclass
class DevicesCommand:
    """Discover connected cameras, robot arms, and input devices."""

    verbose: bool = False

    def run(self) -> None:
        from limb.discovery import discover_devices

        discover_devices(verbose=self.verbose)


@dataclass
class ReplayCommand:
    """Replay a recorded episode on hardware for verification.

    Streams joint commands from a recorded episode to the physical robot.
    Useful for checking recording quality before conversion.
    """

    episode_dir: str = ""
    config_path: Tuple[str, ...] = ("configs/yam_gello_bimanual.yaml",)
    speed: float = 1.0
    log_level: str = "INFO"

    def run(self) -> None:
        from limb.replay import replay_episode

        replay_episode(
            episode_dir=self.episode_dir,
            config_path=[os.path.expanduser(x) for x in self.config_path],
            speed=self.speed,
            log_level=self.log_level,
        )


@dataclass
class ConvertCommand:
    """Convert raw recordings to LeRobot v2.1 dataset format."""

    input_dir: str = ""
    output_dir: str = ""
    task: Optional[str] = None
    robot_type: str = "yam"
    fps: int = 30
    success_only: bool = False
    push_to_hub: Optional[str] = None

    def run(self) -> None:
        from scripts.data.convert_to_lerobot import Args, main

        main(
            Args(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                task=self.task,
                robot_type=self.robot_type,
                fps=self.fps,
                success_only=self.success_only,
                push_to_hub=self.push_to_hub,
            )
        )


@dataclass
class VisualizeCommand:
    """Visualize a recorded episode with Rerun."""

    episode_dir: str = ""

    def run(self) -> None:
        from scripts.data.visualize_episode import Args, main

        main(Args(episode_dir=self.episode_dir))


Command = Union[
    TeleopCommand,
    RecordCommand,
    DevicesCommand,
    ReplayCommand,
    ConvertCommand,
    VisualizeCommand,
]


def cli_main() -> None:
    """Entry point for the `limb` CLI."""
    cmd = tyro.cli(
        Command,
        prog="limb",
        description="limb — minimal, high-frequency control stack for YAM bimanual arms",
    )
    cmd.run()


if __name__ == "__main__":
    cli_main()
