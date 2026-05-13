"""Print live joint readings from a YAM arm on the given CAN channel.

Read-only: talks to DMChainCanInterface directly, so it has no joint-limit
guard and no robot_server thread that can crash silently. Useful for
verifying motor zeros, sanity-checking that the bus is actually streaming,
or watching an arm during back-drive.

Prereq: CAN must already be up (`bash limb/scripts/reset_all_can.sh`) and
nothing else can be holding the channel (kill any `limb teleop` first).

The `!` flag next to a joint reading means it's outside the i2rt-effective
limit (YAML ± 0.1 rad) -- this is the joint range that would trip a
RuntimeError once the limb stack launches the arm. Use `--limits-from` to
load the YAML stack that matches the arm under test; without it, the
script falls back to follower-arm limits from `robot_configs/yam/left.yaml`,
which are TIGHTER than a leader's widened range and will flag valid leader
poses.

Usage:
    # Follower (uses default fallback limits — matches left.yaml).
    uv run python scripts/diagnostics/print_YAM_joints.py --channel can_follow_l

    # Leader (load the same YAML stack the launch config does).
    uv run python scripts/diagnostics/print_YAM_joints.py --channel can_leader_l \\
        --limits-from robot_configs/yam/left.yaml robot_configs/yam/leader_left.yaml
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tyro
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, ReceiveMode
from loguru import logger

from limb.envs.configs.loader import DictLoader

YAM_MOTOR_LIST: list[tuple[int, str]] = [
    (0x01, "DM4340"),
    (0x02, "DM4340"),
    (0x03, "DM4340"),
    (0x04, "DM4310"),
    (0x05, "DM4310"),
    (0x06, "DM4310"),
    (0x07, "DM4310"),
]

# i2rt's `_check_current_qpos_in_joint_limits` widens the YAML's `joint_limits`
# by ±buffer_rad before tripping (dependencies/i2rt/.../motor_chain_robot.py:206).
I2RT_BUFFER_RAD = 0.1

# Fallback when --limits-from is empty. Mirrors robot_configs/yam/left.yaml
# (follower) + the i2rt buffer. Leader arms use a wider range -- load the
# leader overlay via --limits-from to avoid spurious `!` flags.
FOLLOWER_FALLBACK_LIMITS: list[tuple[float, float]] = [
    (-2.09 - I2RT_BUFFER_RAD, 3.14 + I2RT_BUFFER_RAD),  # J1
    (0.00 - I2RT_BUFFER_RAD,  3.14 + I2RT_BUFFER_RAD),  # J2
    (0.05 - I2RT_BUFFER_RAD,  3.14 + I2RT_BUFFER_RAD),  # J3
    (-1.35 - I2RT_BUFFER_RAD, 1.35 + I2RT_BUFFER_RAD),  # J4
    (-1.50 - I2RT_BUFFER_RAD, 1.50 + I2RT_BUFFER_RAD),  # J5
    (-2.00 - I2RT_BUFFER_RAD, 2.00 + I2RT_BUFFER_RAD),  # J6
]


@dataclass
class Args:
    channel: str = "can_leader_l"
    hz: float = 20.0
    # Stacked YAML paths to load `joint_limits` from, applied left-to-right
    # (later files override). Pass the same list a launch config would (e.g.
    # `left.yaml leader_left.yaml` for the left leader). If empty, the script
    # uses FOLLOWER_FALLBACK_LIMITS (follower-arm limits from left.yaml).
    limits_from: tuple[str, ...] = ()


def _load_limits(paths: tuple[str, ...]) -> list[tuple[float, float]]:
    """Load `joint_limits` from a stacked YAML config and add the i2rt buffer.

    Mirrors the limit logic in `MotorChainRobot._check_current_qpos_in_joint_limits`:
    the YAML lists the soft limit, runtime widens by ±I2RT_BUFFER_RAD on each side.
    """
    cfg = DictLoader.load(list(paths))
    if "joint_limits" not in cfg:
        raise ValueError(
            f"--limits-from {list(paths)} produced no `joint_limits` key. "
            "Did you forget the base YAML (e.g. left.yaml before leader_left.yaml)?"
        )
    raw = cfg["joint_limits"]
    return [(float(lo) - I2RT_BUFFER_RAD, float(hi) + I2RT_BUFFER_RAD) for lo, hi in raw]


def main(args: Args) -> None:
    if args.limits_from:
        joint_limits = _load_limits(args.limits_from)
        logger.info(f"Loaded joint_limits from {list(args.limits_from)} (+/-{I2RT_BUFFER_RAD} rad i2rt buffer)")
    else:
        joint_limits = FOLLOWER_FALLBACK_LIMITS
        logger.info(
            "No --limits-from given; using follower fallback limits from left.yaml. "
            "Pass --limits-from for a leader to silence spurious `!` flags."
        )
    logger.info(f"Opening {args.channel}")
    chain = DMChainCanInterface(
        motor_list=YAM_MOTOR_LIST,
        motor_offset=[0] * 7,
        motor_direction=[1] * 7,
        channel=args.channel,
        motor_chain_name=f"watch_{args.channel}",
        receive_mode=ReceiveMode("p16"),
    )
    period = 1.0 / args.hz
    logger.info(f"Reading at {args.hz:.0f} Hz — Ctrl+C to stop")
    logger.info("J{i} is 1-indexed (matches the YAML/URDF); i2rt error messages are 0-indexed.")
    try:
        while True:
            q = np.array([s.pos for s in chain.read_states()])
            cells = []
            for i, v in enumerate(q[:6]):
                lo, hi = joint_limits[i]
                flag = "!" if v < lo or v > hi else " "
                cells.append(f"J{i + 1}:{v:+.4f}{flag}")
            print("  ".join(cells) + f"  grip:{q[6]:+.4f}", end="\r", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print()
        logger.info("stopped")


if __name__ == "__main__":
    main(tyro.cli(Args))
