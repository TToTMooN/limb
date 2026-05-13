"""Bring up a YAM leader arm in zero-torque + gravity-comp mode, override the
correction terms on one or more joints, record all joint positions and
reported motor torques while you backdrive the arm, then plot each targeted
joint's q-vs-torque curve.

The baseline correction (scale / offset / slope / damping per joint) mirrors
``robot_configs/yam/leader_left.yaml``.  Pass parallel lists ``--joint-indices``
(0-indexed) and ``--alphas`` / ``--betas`` / ``--slopes`` / ``--dampings`` to
override the listed joints; any unlisted joint keeps its baseline value so it
holds while you tune.

If a value list is non-empty its length must match ``--joint-indices``.  Pass
an empty list (the default) for any knob you don't want to override.

Usage:
    # Hold-test current calibration on J4 with leader_left baseline.
    uv run scripts/diagnostics/tune_gravity_comp.py

    # Re-tune J4 with new constants.
    uv run scripts/diagnostics/tune_gravity_comp.py \\
        --joint-indices 3 --alphas 1.20 --betas 0.10 --slopes 0.40 --dampings 0.20

    # Tune J3 from scratch (scale only).
    uv run scripts/diagnostics/tune_gravity_comp.py --joint-indices 2 --alphas 1.10

    # Tune J3 and J4 simultaneously — both alpha, J4 also gets new beta.
    uv run scripts/diagnostics/tune_gravity_comp.py \\
        --joint-indices 2 3 --alphas 1.10 1.20 --betas 0.0 0.10

    # Switch arm.
    uv run scripts/diagnostics/tune_gravity_comp.py --channel can_leader_r

CAN must already be up (`bash limb/scripts/reset_all_can.sh`).
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, ReceiveMode
from loguru import logger

from limb.robots.yam_motor_chain_robot import YamMotorChainRobot

DEFAULT_XML_PATH = "assets/yam/yam_leaderhandle_autel.xml"

# Baseline mirrors robot_configs/yam/leader_left.yaml.  --alpha/--beta/--slope
# /--damping override the corresponding slot at --joint-index; all other slots
# stay at these values so untargeted joints hold their pose during the sweep.
BASELINE_SCALE = (1.08, 1.08, 1.10, 1.17, 1.08, 1.08, 1.0)
BASELINE_OFFSET = (0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0)
BASELINE_SLOPE = (0.0, 0.0, 0.05, 0.35, 0.0, 0.0, 0.0)
BASELINE_DAMPING = (0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0)


@dataclass
class Args:
    channel: str = "can_leader_l"
    sample_hz: float = 200.0
    # Joints to override (0-indexed; 0=J1 ... 5=J6, 6=gripper).  First entry
    # is used as the focus joint for the q-vs-tau plot.  Untouched joints
    # keep their baseline values.
    joint_indices: tuple[int, ...] = (3,)
    # Per-joint override lists.  Each list is either empty (= keep baseline
    # for every listed joint on this knob) or has length == len(joint_indices)
    # giving one value per listed joint.  Mixing is fine, e.g. alphas for
    # both joints but betas for only J4 -> pass `betas` of length == len.
    #
    # Effective per-joint correction:
    #   tau[j] = alpha * tau_model[j] + beta + slope*q[j] - damping*qdot[j]
    alphas: tuple[float, ...] = ()
    betas: tuple[float, ...] = ()
    slopes: tuple[float, ...] = ()
    dampings: tuple[float, ...] = ()
    xml_path: str = DEFAULT_XML_PATH
    out_path: Optional[Path] = None
    npz_path: Optional[Path] = None


def _validate(args: Args) -> None:
    if not args.joint_indices:
        raise ValueError("--joint-indices must list at least one joint")
    for j in args.joint_indices:
        if not (0 <= j <= 6):
            raise ValueError(f"joint index {j} out of range [0, 6]")
    n = len(args.joint_indices)
    for name, vals in (("alphas", args.alphas), ("betas", args.betas),
                       ("slopes", args.slopes), ("dampings", args.dampings)):
        if vals and len(vals) != n:
            raise ValueError(
                f"--{name} has {len(vals)} entries but --joint-indices has {n}; "
                f"pass either nothing (keep baseline) or one value per joint."
            )


def _build_arrays(args: Args) -> tuple[list[float], list[float], list[float], list[float]]:
    scale = list(BASELINE_SCALE)
    offset = list(BASELINE_OFFSET)
    slope = list(BASELINE_SLOPE)
    damping = list(BASELINE_DAMPING)
    for k, j in enumerate(args.joint_indices):
        if args.alphas:
            scale[j] = args.alphas[k]
        if args.betas:
            offset[j] = args.betas[k]
        if args.slopes:
            slope[j] = args.slopes[k]
        if args.dampings:
            damping[j] = args.dampings[k]
    return scale, offset, slope, damping


def _build_robot(args: Args) -> YamMotorChainRobot:
    motor_chain = DMChainCanInterface(
        motor_list=[
            [0x01, "DM4340"],
            [0x02, "DM4340"],
            [0x03, "DM4340"],
            [0x04, "DM4310"],
            [0x05, "DM4310"],
            [0x06, "DM4310"],
            [0x07, "DM4310"],
        ],
        motor_offset=[0, 0, 0, 0, 0, 0, 0],
        motor_direction=[1, 1, 1, 1, 1, 1, 1],
        channel=args.channel,
        motor_chain_name=f"yam_zero_g_{args.channel}",
        receive_mode=ReceiveMode("p16"),
    )
    scale, offset, slope, damping = _build_arrays(args)
    return YamMotorChainRobot(
        motor_chain=motor_chain,
        xml_path=args.xml_path,
        gripper_index=6,
        gripper_limits=[0.0, -4.0],
        limit_gripper_force=10.0,
        kp=[80, 80, 80, 10, 10, 10, 2],
        kd=[5, 5, 5, 1.5, 1.5, 1.5, 0.1],
        gravity_comp_factor=1.0,
        gravity_comp_per_joint_scale=scale,
        gravity_comp_per_joint_offset=offset,
        gravity_comp_per_joint_slope=slope,
        gravity_comp_per_joint_damping=damping,
        temp_record_flag=False,
    )


def _plot(q: np.ndarray, tau: np.ndarray, joint_idx: int, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(q, tau, c=np.arange(len(q)), cmap="viridis", s=4)
    ax.set_xlabel(f"joint {joint_idx + 1} position [rad]")
    ax.set_ylabel(f"joint {joint_idx + 1} reported motor torque [Nm]")
    ax.set_title(f"YAM leader  —  J{joint_idx + 1} q vs torque  ({len(q)} samples)")
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("sample index (time)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote plot -> {out_path}")


def main(args: Args) -> None:
    _validate(args)
    if 6 in args.joint_indices:
        logger.warning("joint 6 is the gripper; inverse dynamics returns 0 for it so tuning is a no-op.")

    primary = args.joint_indices[0]
    suffix = "_".join(f"j{j + 1}" for j in args.joint_indices)
    out_path = args.out_path or Path(f"recordings/zero_g_{suffix}.png")
    npz_path = args.npz_path or Path(f"recordings/zero_g_{suffix}.npz")

    scale, offset, slope, damping = _build_arrays(args)
    logger.info(f"Connecting to {args.channel} with XML {args.xml_path}")
    for j in args.joint_indices:
        logger.info(f"Target J{j + 1} (0-idx {j}): "
                    f"alpha={scale[j]:.4f}  beta={offset[j]:.4f}  "
                    f"slope={slope[j]:.4f}  damping={damping[j]:.4f}")

    robot = _build_robot(args)
    robot.zero_torque_mode()
    logger.info("Robot in zero-torque mode (gravity comp on). Move it around. Ctrl+C to stop.")

    stop = False

    def _sigint(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    period = 1.0 / args.sample_hz
    ts, qs, taus = [], [], []
    t0 = time.monotonic()

    try:
        next_t = time.monotonic()
        while not stop:
            obs = robot.get_observations()
            # joint_pos is (6,) arm, gripper_pos is (1,); concatenate to (7,).
            qpos = np.concatenate([obs["joint_pos"], obs["gripper_pos"]])
            qeff = obs["joint_eff"]  # (7,)
            ts.append(time.monotonic() - t0)
            qs.append(qpos.copy())
            taus.append(qeff.copy())

            next_t += period
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Fell behind; resync.
                next_t = time.monotonic()
    finally:
        logger.info(f"Captured {len(qs)} samples. Soft-releasing arm...")
        try:
            robot.soft_release(duration_s=1.5, steps=40)
        except Exception as e:
            logger.warning(f"soft_release failed: {e}")
        robot.close()

    if not qs:
        logger.warning("No samples recorded.")
        return

    q_arr = np.asarray(qs)  # (N, 7)
    tau_arr = np.asarray(taus)  # (N, 7)
    t_arr = np.asarray(ts)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        t=t_arr,
        q=q_arr,
        tau=tau_arr,
        joint_indices=np.asarray(args.joint_indices, dtype=np.int64),
        scale=np.asarray(scale),
        offset=np.asarray(offset),
        slope=np.asarray(slope),
        damping=np.asarray(damping),
    )
    logger.info(f"Wrote raw samples -> {npz_path}")

    # One plot per targeted joint.  Single-joint case keeps the previous
    # filename (out_path); multi-joint case derives sibling filenames.
    for k, j in enumerate(args.joint_indices):
        if k == 0:
            per_path = out_path
        else:
            per_path = out_path.with_name(out_path.stem + f"_j{j + 1}" + out_path.suffix)
        _plot(q_arr[:, j], tau_arr[:, j], j, per_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
