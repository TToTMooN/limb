"""
Standalone GELLO (Dynamixel) input diagnostic — no robot hardware needed.

Tests:
  1. Dynamixel USB serial connection (or network via --host)
  2. Joint position reading from all servos
  3. Sign-corrected angles for left/right arms
  4. (Optional) Full agent loop with PrintRobot (no real YAM)

Usage:
  # Raw positions via direct USB (default: bimanual, /dev/ttyUSB0):
  uv run scripts/test_gello_input.py

  # Raw positions via network (R1 Lite Teleop over Ethernet):
  uv run scripts/test_gello_input.py --host 10.42.0.1

  # Single arm, custom port:
  uv run scripts/test_gello_input.py --port /dev/r1litet_usb --single-arm

  # Full agent loop with fake follower robot (no real YAM):
  uv run scripts/test_gello_input.py --with-agent
  uv run scripts/test_gello_input.py --with-agent --host 10.42.0.1

Prerequisites:
  - Direct USB: R1 Lite Teleop connected via USB + dynamixel-sdk
  - Network: gello_position_server.py running on the R1 Lite Teleop device
"""

import argparse
import time

import numpy as np
from rich.live import Live
from rich.table import Table


def _fmt(arr: np.ndarray, f: str = "+.3f") -> str:
    return " ".join(f"{v:{f}}" for v in arr)


def _make_raw_table(
    reader: "DynamixelReader",  # noqa: F821
    n_left: int,
    bimanual: bool,
    read_count: int,
    error_count: int,
) -> Table:
    """Build a Rich table showing raw Dynamixel joint positions."""
    table = Table(title="GELLO Dynamixel Input", show_header=True, header_style="bold")
    table.add_column("Side", width=6)
    table.add_column("J1", width=8)
    table.add_column("J2", width=8)
    table.add_column("J3", width=8)
    table.add_column("J4", width=8)
    table.add_column("J5", width=8)
    table.add_column("J6", width=8)

    try:
        pos = reader.get_joint_positions()
        read_count += 1
    except Exception as e:
        table.add_row("", f"[red]Read error: {e}[/]", "", "", "", "", "")
        return table

    left = pos[:n_left]
    table.add_row(
        "left",
        *[f"{v:+.3f}" for v in left],
    )

    if bimanual:
        right = pos[n_left:]
        table.add_row(
            "right",
            *[f"{v:+.3f}" for v in right],
        )

    table.add_section()
    table.add_row(
        "",
        f"[dim]reads: {read_count}[/]",
        f"[dim]errors: {error_count}[/]",
        "",
        "",
        "",
        "",
    )

    return table


def test_raw_input(port: str, baudrate: int, bimanual: bool, host: str | None = None, network_port: int = 5555) -> None:
    """Print raw Dynamixel joint positions with rich Live display."""
    if host is not None:
        from limb.devices.network_dynamixel_reader import NetworkDynamixelReader

        print(f"Connecting to {host}:{network_port} (network mode, bimanual={bimanual})...")
        reader = NetworkDynamixelReader(host=host, port=network_port)
    else:
        from limb.devices.dynamixel_reader import DynamixelReader

        left_ids = [1, 2, 3, 4, 5, 6]
        right_ids = [7, 8, 9, 10, 11, 12]
        left_signs = [1, 1, -1, -1, -1, 1]
        right_signs = [1, 1, -1, -1, -1, 1]

        all_ids = left_ids + (right_ids if bimanual else [])
        all_signs = left_signs + (right_signs if bimanual else [])

        print(f"Connecting to {port} (baudrate={baudrate}, bimanual={bimanual})...")
        reader = DynamixelReader(
            port=port,
            motor_ids=all_ids,
            joint_signs=all_signs,
            baudrate=baudrate,
        )

    print("Connected! Reading joint positions... (Ctrl+C to stop)\n")

    read_count = 0
    error_count = 0
    n_left = 6

    with Live(
        _make_raw_table(reader, n_left, bimanual, read_count, error_count),
        refresh_per_second=20,
    ) as live:
        try:
            while True:
                try:
                    live.update(_make_raw_table(reader, n_left, bimanual, read_count, error_count))
                    read_count += 1
                except Exception:
                    error_count += 1
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            reader.close()

    print("Stopped.")


def _make_agent_table(action: dict, step: int) -> Table:
    """Build a Rich table showing the agent output."""
    table = Table(title=f"GELLO Agent — step {step}", show_header=True, header_style="bold")
    table.add_column("Side", width=6)
    table.add_column("Arm Joints (6)")
    table.add_column("Gripper", width=8)

    for side in ("left", "right"):
        if side not in action:
            continue
        j = action[side]["pos"]
        table.add_row(
            side,
            _fmt(j[:6], "+.3f"),
            f"{j[6]:.3f}",
        )

    return table


def test_with_agent(port: str, baudrate: int, bimanual: bool, host: str | None = None, network_port: int = 5555) -> None:
    """Run GELLO agent with PrintRobot — no real YAM hardware needed."""
    from limb.agents.teleoperation.yam_gello_agent import YamGelloAgent
    from limb.robots.robot import PrintRobot

    print("=== GELLO Agent Test (no real robot) ===")
    mode = f"network ({host}:{network_port})" if host else f"USB ({port})"
    print(f"Mode: {mode}")
    print("The agent reads Dynamixel positions and computes joint targets.")
    print("Move the GELLO leader arms to see the output change.\n")

    agent = YamGelloAgent(
        port=port,
        baudrate=baudrate,
        host=host,
        network_port=network_port,
        bimanual=bimanual,
    )

    left_robot = PrintRobot(num_dofs=7, dont_print=True)
    right_robot = PrintRobot(num_dofs=7, dont_print=True) if bimanual else None

    dummy_action: dict = {"left": {"pos": np.zeros(7)}}
    if bimanual:
        dummy_action["right"] = {"pos": np.zeros(7)}

    with Live(_make_agent_table(dummy_action, 0), refresh_per_second=20) as live:
        try:
            step = 0
            while True:
                obs: dict = {
                    "left": {"joint_pos": left_robot.get_joint_pos()},
                    "timestamp": time.time(),
                }
                if bimanual and right_robot is not None:
                    obs["right"] = {"joint_pos": right_robot.get_joint_pos()}

                action = agent.act(obs)

                left_robot.command_joint_pos(action["left"]["pos"])
                if bimanual and "right" in action and right_robot is not None:
                    right_robot.command_joint_pos(action["right"]["pos"])

                step += 1
                live.update(_make_agent_table(action, step))
                time.sleep(1.0 / 100.0)
        except KeyboardInterrupt:
            pass
        finally:
            agent.close()

    print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GELLO Dynamixel input for YAM teleop")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Dynamixel USB serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=4_000_000, help="Serial baudrate (default: 4000000)")
    parser.add_argument("--host", default=None, help="R1 Lite Teleop IP for network mode (e.g. 10.42.0.1)")
    parser.add_argument("--network-port", type=int, default=5555, help="TCP port for network mode (default: 5555)")
    parser.add_argument("--single-arm", action="store_true", help="Single arm mode (left only, IDs 1-6)")
    parser.add_argument("--with-agent", action="store_true", help="Run full agent loop with PrintRobot (no real YAM)")
    args = parser.parse_args()

    bimanual = not args.single_arm

    if args.with_agent:
        test_with_agent(args.port, args.baudrate, bimanual, host=args.host, network_port=args.network_port)
    else:
        test_raw_input(args.port, args.baudrate, bimanual, host=args.host, network_port=args.network_port)
