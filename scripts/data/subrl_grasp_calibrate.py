"""Calibrate the VERIFIER (sub-task start state + RL success) from human GRASP demos.

The coding-agent verifier has two jobs the demos ground directly (user 2026-07-06):
  1. identify the bottleneck sub-task's STARTING POINT — where the gripper is when a
     real grasp begins (EE xy/z band, gripper open) = the state the reset must restore
     and the state the RL-entry gate should verify;
  2. score RL rollouts — grasp SUCCESS thresholds (held width band, gripper effort,
     lift height).

Usage (episodes labeled SUCCESS are the grasp demos in the mixed session):
    uv run python scripts/data/subrl_grasp_calibrate.py \
        --episodes-dir recordings/<session> --label success

Per grasp event (gripper open -> held): z/xy at close, the APPROACH point (last tick
the gripper was still fully open above the close height), held width + gripper effort
during the hold, and the lift height reached. All FK in the runtime's exact frame.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import tyro
from loguru import logger

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from limb.agents.policy_learning.subtask.fk import EEPoseInjector  # noqa: E402


@dataclass
class Args:
    episodes_dir: str
    arm: str = "right"
    label: str = "success"
    open_width: float = 0.85
    held_min: float = 0.02
    held_max: float = 0.15
    confirm_frames: int = 8
    table_z: float = 0.065
    fps: float = 30.0


def _events(gp: np.ndarray, a: Args) -> list[tuple[int, int]]:
    """(open_end, grasp_idx) pairs: index of the last confirmed-open tick before each
    debounced open->held transition, plus the held onset."""
    def state(w):
        if w >= a.open_width:
            return "open"
        if a.held_min <= w <= a.held_max:
            return "held"
        return "mid"

    # A closing gripper passes THROUGH the mid band, so entry into 'held' arrives from
    # 'mid' — grasp = any debounced entry into 'held' after the gripper has been open.
    events, cur, cand, cand_n = [], state(gp[0]), None, 0
    last_open = 0 if state(gp[0]) == "open" else None
    for i in range(1, len(gp)):
        s = state(gp[i])
        if s == cur:
            if cur == "open":
                last_open = i
            cand, cand_n = None, 0
            continue
        if s != cand:
            cand, cand_n = s, 1
        else:
            cand_n += 1
        if cand_n >= a.confirm_frames:
            cur = cand
            cand, cand_n = None, 0
            j = i - a.confirm_frames + 1
            if cur == "held" and last_open is not None:
                events.append((last_open, j))
                last_open = None                     # one event per open->...->held arc
    return events


def main(args: Args) -> None:
    root = pathlib.Path(args.episodes_dir)
    eps = sorted(root.glob("episode_*"))
    if args.label:
        eps = [e for e in eps if (e / args.label.upper()).exists()]
    if not eps:
        raise SystemExit(f"no '{args.label}'-labeled episode_* under {root}")
    fk = EEPoseInjector(sides=[args.arm])

    rows = []
    for ep in eps:
        states = np.load(ep / f"{args.arm}_states.npz")
        jp = states["joint_pos"]
        gp = states["gripper_pos"].reshape(len(jp), -1)[:, 0]
        eff = np.abs(states["joint_eff"][:, 6]) if "joint_eff" in states and states["joint_eff"].shape[1] >= 7 else None
        poses = np.stack([fk.ee_pose(q) for q in jp])
        z, x, y = poses[:, 6], poses[:, 4], poses[:, 5]
        for open_end, g in _events(gp, args):
            hold = slice(g, min(len(z), g + int(3 * args.fps)))   # ~3 s after close
            # approach point: where the descent-to-grasp started (last open tick that is
            # >= 3 cm above the close height)
            appr = open_end
            for k in range(open_end, g):
                if gp[k] >= args.open_width and z[k] >= z[g] + 0.03:
                    appr = k
            rows.append(dict(
                ep=ep.name, grasp_z=float(z[g]), grasp_xy=(float(x[g]), float(y[g])),
                appr_z=float(z[appr]), appr_xy=(float(x[appr]), float(y[appr])),
                held_w_p50=float(np.percentile(gp[hold], 50)),
                held_eff_p50=(float(np.percentile(eff[hold], 50)) if eff is not None else float("nan")),
                lift=float(z[hold].max() - z[g]),
            ))

    if not rows:
        raise SystemExit("no open->held grasp events found")

    logger.info("{} grasp events from {} episode(s)", len(rows), len(eps))
    for r in rows:
        logger.info("  {}: close z={:.3f} xy=({:.3f},{:.3f})  approach z={:.3f}  "
                    "held w={:.3f} eff={:.3f}  lift=+{:.3f}",
                    r["ep"], r["grasp_z"], *r["grasp_xy"], r["appr_z"],
                    r["held_w_p50"], r["held_eff_p50"], r["lift"])

    def pct(k, q):
        return float(np.nanpercentile([r[k] for r in rows], q))

    tz = args.table_z
    logger.info("\n---- verifier / start-state suggestions (table_z={:.3f}) ----", tz)
    logger.info("GRASP z band: p5/p50/p95 = {:.3f}/{:.3f}/{:.3f}  (place_z capture band is "
                "z <= table_z+0.10 = {:.3f} — every real close must sit BELOW it)",
                pct("grasp_z", 5), pct("grasp_z", 50), pct("grasp_z", 95), tz + 0.10)
    logger.info("START point (approach, gripper open): z p5/p50/p95 = {:.3f}/{:.3f}/{:.3f} "
                "-> selector approach_z band and the reset RAISE height should cover this",
                pct("appr_z", 5), pct("appr_z", 50), pct("appr_z", 95))
    xs = [r["grasp_xy"][0] for r in rows]; ys = [r["grasp_xy"][1] for r in rows]
    logger.info("grasp xy span: x [{:.3f},{:.3f}]  y [{:.3f},{:.3f}]  (compare PICKUP_REGION)",
                min(xs), max(xs), min(ys), max(ys))
    logger.info("HELD width p5/p50/p95 = {:.3f}/{:.3f}/{:.3f}  (verifier band [0.02, 0.15])",
                pct("held_w_p50", 5), pct("held_w_p50", 50), pct("held_w_p50", 95))
    logger.info("HELD gripper effort p5/p50/p95 = {:.3f}/{:.3f}/{:.3f}  (LOAD_HOLD=0.25)",
                pct("held_eff_p50", 5), pct("held_eff_p50", 50), pct("held_eff_p50", 95))
    logger.info("LIFT reached p5/p50 = {:.3f}/{:.3f} m above the close (verifier lift_m=0.05)",
                pct("lift", 5), pct("lift", 50))


if __name__ == "__main__":
    main(tyro.cli(Args))
