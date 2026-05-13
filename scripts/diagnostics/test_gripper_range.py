"""Live gripper readout for a YAM arm — used to calibrate gripper_limits.

Spins up a single YAM arm on the specified CAN channel, puts it in
zero-torque + gravity-comp mode (operator can backdrive freely), and
prints both:

* ``raw`` — the raw motor 0x07 position in radians, unclipped.  Compare
  this against the ``[closed, open]`` values in
  ``robot_configs/yam/{left,right}.yaml``'s ``gripper_limits``.
* ``norm`` — the normalized [0, 1] value derived from those limits.
  This is what the rest of the stack sees as ``gripper_pos`` in obs and
  what ``YamYamBilateralAgent`` forwards leader -> follower.

Typical use (measuring the physical gripper open extreme):

    1. Start this script on the arm you want to calibrate.
    2. Push the gripper jaws (or trigger handle) to its physical "open"
       hardstop by hand.
    3. Read the peak ``raw`` value.  That is your hardware's true open
       motor position.  Set ``gripper_limits[1]`` in the arm's YAML
       slightly less negative (a little conservative) than that value.
    4. Squeeze fully closed and confirm ``raw`` reaches ~0.0 (or your
       configured closed limit).

The arm boots into ``zero_torque_mode`` so the operator can manipulate
the trigger freely; arm joints are also backdrivable.

Usage:

    uv run scripts/diagnostics/test_gripper_range.py --channel can_follow_l
    uv run scripts/diagnostics/test_gripper_range.py --channel can_leader_l --hz 20

CAN must already be up (`bash limb/scripts/reset_all_can.sh`).
Ctrl+C to exit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tyro
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, ReceiveMode
from loguru import logger

from limb.robots.yam_motor_chain_robot import YamMotorChainRobot

XML_PATH = "dependencies/i2rt/i2rt/robot_models/yam/yam.xml"


@dataclass
class Args:
    channel: str = "can_follow_l"
    """SocketCAN channel for the arm under test (e.g. can_follow_l, can_leader_l)."""
    hz: float = 10.0
    """Print rate in Hz."""
    closed_limit: float = 0.0
    """Closed-side gripper limit in raw motor radians (for the normalization)."""
    open_limit: float = -5.2
    """Open-side gripper limit in raw motor radians (for the normalization).
    Match the value in robot_configs/yam/left.yaml's gripper_limits[1]."""


def main(args: Args) -> None:
    motor_chain = DMChainCanInterface(
        motor_list=[
            [0x01, "DM4340"],
            [0x02, "DM4340"],
            [0x03, "DM4340"],
            [0x04, "DM4310"],
            [0x05, "DM4310"],
            [0x06, "DM4310"],
            [0x07, "DM4310"],  # gripper
        ],
        motor_offset=[0, 0, 0, 0, 0, 0, 0],
        motor_direction=[1, 1, 1, 1, 1, 1, 1],
        channel=args.channel,
        motor_chain_name=f"yam_gripper_range_{args.channel}",
        receive_mode=ReceiveMode("p16"),
    )

    robot = YamMotorChainRobot(
        motor_chain=motor_chain,
        xml_path=XML_PATH,
        gripper_index=6,
        gripper_limits=[args.closed_limit, args.open_limit],
        limit_gripper_force=10.0,
        kp=[80, 80, 80, 10, 10, 10, 2],
        kd=[5, 5, 5, 1.5, 1.5, 1.5, 0.1],
        gravity_comp_factor=1.0,
        temp_record_flag=False,
    )

    # Make every joint freely backdrivable so the operator can move the arm
    # AND squeeze the gripper without fighting any PD term.  Gravity comp
    # stays on for the arm joints.
    robot.zero_torque_mode()

    logger.info(
        "Gripper-range readout on {} | configured limits: closed={:.3f}, open={:.3f}",
        args.channel,
        args.closed_limit,
        args.open_limit,
    )
    logger.info(
        "Push the gripper through its full mechanical range (squeeze fully closed, "
        "then fully open) and watch the `raw` and `norm` columns."
    )
    logger.info("Press Ctrl+C to exit.")
    print()
    print(f"{'raw [rad]':>12}  {'norm [0..1]':>12}  {'min raw':>10}  {'max raw':>10}")
    print("-" * 52)

    dt = 1.0 / args.hz
    raw_min = float("inf")
    raw_max = float("-inf")

    try:
        while True:
            obs = robot.get_observations()
            # gripper_pos is post-remap (normalized [0, 1]).  We also want the
            # raw motor reading — re-extract it via the remapper inverse.
            norm = float(np.asarray(obs["gripper_pos"]).reshape(-1)[0])
            raw = norm * (args.open_limit - args.closed_limit) + args.closed_limit
            raw_min = min(raw_min, raw)
            raw_max = max(raw_max, raw)
            print(
                f"{raw:12.4f}  {norm:12.4f}  {raw_min:10.4f}  {raw_max:10.4f}",
                end="\r",
                flush=True,
            )
            time.sleep(dt)
    except KeyboardInterrupt:
        print()
        logger.info(
            "Observed raw range: [{:.4f}, {:.4f}] (span = {:.4f} rad)",
            raw_min,
            raw_max,
            raw_max - raw_min,
        )
        logger.info(
            "Observed normalized range: [{:.4f}, {:.4f}]",
            (raw_min - args.closed_limit) / (args.open_limit - args.closed_limit),
            (raw_max - args.closed_limit) / (args.open_limit - args.closed_limit),
        )
        logger.info(
            "If norm doesn't span ~[0.0, 1.0]: adjust open_limit (or YAML's "
            "gripper_limits[1]) to match the raw extreme you actually reached."
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
