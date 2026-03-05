"""
Standalone VR input diagnostic — no robot hardware needed.

Tests:
  1. XRoboToolkit SDK connection
  2. Controller pose streaming
  3. Grip / trigger values
  4. (Optional) IK solving + Viser visualization with PrintRobot

Usage:
  # Just print VR input (no Viser, no robot):
  uv run scripts/test_vr_input.py

  # Also run IK + Viser visualization (safe, uses PrintRobot):
  uv run scripts/test_vr_input.py --with-ik

Prerequisites:
  - XRoboToolkit PC Service running
  - Pico headset connected and streaming
"""

import argparse
import time

import numpy as np
from rich.live import Live
from rich.table import Table


def _fmt(arr: np.ndarray, f: str = "+.3f") -> str:
    return " ".join(f"{v:{f}}" for v in arr)


def _make_raw_table(xr, zero_count: int) -> Table:
    """Build a Rich table showing raw VR input."""
    table = Table(title="VR Input", show_header=True, header_style="bold")
    table.add_column("Side", width=6)
    table.add_column("Status", width=8)
    table.add_column("Position (xyz)")
    table.add_column("Quaternion (xyzw)")
    table.add_column("Grip", width=6)
    table.add_column("Trigger", width=8)

    for side in ("left", "right"):
        pose = xr.get_pose(f"{side}_controller")
        grip = xr.get_grip(side)
        trigger = xr.get_trigger(side)
        pos = pose[:3]
        quat = pose[3:]
        status = "[green]ACTIVE[/]" if grip > 0.9 else "[dim]idle[/]"
        table.add_row(
            side,
            status,
            _fmt(pos),
            _fmt(quat),
            f"{grip:.2f}",
            f"{trigger:.2f}",
        )

    buttons = {b: xr.get_button(b) for b in ("A", "B", "X", "Y")}
    pressed = [f"[bold]{b}[/]" for b, v in buttons.items() if v]
    btn_str = " ".join(pressed) if pressed else "[dim]none[/]"
    table.add_section()
    table.add_row("", "", f"Buttons: {btn_str}", "", "", "")

    if zero_count >= 5:
        table.add_row(
            "", "[bold red]WARN[/]", "[red]All values zero — check PC Service / Pico[/]", "", "", ""
        )

    return table


def test_raw_input() -> None:
    """Print raw VR controller data with rich Live display."""
    from limb.devices.xr_client import XrClient

    xr = XrClient()

    zero_count = 0
    with Live(_make_raw_table(xr, 0), refresh_per_second=10) as live:
        try:
            while True:
                all_zero = True
                for side in ("left", "right"):
                    pose = xr.get_pose(f"{side}_controller")
                    pos = pose[:3]
                    quat = pose[3:]
                    if np.any(np.abs(pos) > 1e-6) or np.any(np.abs(quat) > 1e-6):
                        all_zero = False
                        break

                if all_zero:
                    zero_count += 1
                else:
                    zero_count = 0

                live.update(_make_raw_table(xr, zero_count))
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            xr.close()

    print("Stopped.")


def _make_ik_table(agent, action, step: int) -> Table:
    """Build a Rich table showing IK teleop status."""
    table = Table(title=f"VR Teleop — step {step}", show_header=True, header_style="bold")
    table.add_column("Side", width=6)
    table.add_column("Status", width=8)
    table.add_column("Grip", width=6)
    table.add_column("Joints")
    table.add_column("Gripper", width=8)

    for side in ("left", "right"):
        if side not in action:
            continue
        j = action[side]["pos"]
        grip = agent.xr_client.get_grip(side)
        active = agent.active.get(side, False)
        status = "[green]ACTIVE[/]" if active else "[dim]idle[/]"
        table.add_row(
            side,
            status,
            f"{grip:.2f}",
            _fmt(j[:6], "+.2f"),
            f"{j[6]:.2f}",
        )

    return table


def test_with_ik() -> None:
    """Run VR teleop with IK solving and Viser visualization, but no real robot."""
    from limb.agents.teleoperation.yam_vr_agent import YamVrAgent
    from limb.robots.robot import PrintRobot

    print("=== VR Teleop Test with IK (no real robot) ===")
    print("Open http://localhost:8080 in browser to see Viser visualization.")
    print("Squeeze grip to activate arm, move hand, squeeze trigger for gripper.\n")

    agent = YamVrAgent(
        bimanual=True,
        right_arm_extrinsic={"position": [0.0, -0.61, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        scale_factor=1.5,
    )

    left_robot = PrintRobot(num_dofs=7, dont_print=True)
    right_robot = PrintRobot(num_dofs=7, dont_print=True)

    dummy_action = {
        "left": {"pos": np.zeros(7)},
        "right": {"pos": np.zeros(7)},
    }

    with Live(_make_ik_table(agent, dummy_action, 0), refresh_per_second=10) as live:
        try:
            step = 0
            while True:
                obs = {
                    "left": {"joint_pos": left_robot.get_joint_pos()},
                    "right": {"joint_pos": right_robot.get_joint_pos()},
                    "timestamp": time.time(),
                }

                action = agent.act(obs)

                left_robot.command_joint_pos(action["left"]["pos"])
                if "right" in action:
                    right_robot.command_joint_pos(action["right"]["pos"])

                step += 1
                live.update(_make_ik_table(agent, action, step))
                time.sleep(1.0 / 30.0)
        except KeyboardInterrupt:
            pass
        finally:
            agent.xr_client.close()

    print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VR input for YAM teleop")
    parser.add_argument("--with-ik", action="store_true", help="Run full IK + Viser visualization (no real robot)")
    args = parser.parse_args()

    if args.with_ik:
        test_with_ik()
    else:
        test_raw_input()
