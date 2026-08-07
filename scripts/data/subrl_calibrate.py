"""Calibrate SubRL vials-grasp geometry + grasp thresholds from a RECORDED episode.

Rides the existing limb pipelines — the normal DAgger session works as-is:
  1) record with the standard DAgger flow (pause <-> autonomous <-> correcting):
       uv run limb record --config-path configs/yam_dagger_pi0_bimanual.yaml configs/dagger_collection.yaml
     During CORRECTING (leader arms drive the followers), with the RIGHT arm: (a) touch
     the gripper tip on the TABLE TOP in 3-4 spots, (b) sweep the perimeter of the
     intended PICKUP REGION near table height, (c) grasp a vial, squeeze, lift ~10 cm,
     hold ~2 s, (d) release + retract. (Grasps that occur naturally while correcting
     also count as evidence.)
  2) analyze ONLY the human-driven frames:
       uv run python scripts/data/subrl_calibrate.py --episode-dir recordings/<task>/episode_... \
           --phase correcting

Prints: TABLE_Z, PICKUP_REGION (x_min,x_max,y_min,y_max), and gripper-effort threshold
suggestions (LOAD_LOW / LOAD_HOLD) with the supporting evidence. ee poses come from
pinocchio FK on joint_pos (same EEPoseInjector the runtime uses), so the calibrated
frame is IDENTICAL to what the verifier/reset will see.
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
    episode_dir: str
    arm: str = "right"
    phase: str = ""                      # e.g. "correcting" — analyze only frames with this
                                          # phase label (DAgger recordings save phase.npy);
                                          # empty = all frames
    near_table_band_m: float = 0.03      # xy samples within this band above table_z define the region
    # OBSERVATION units (measured gripper_pos is 0-1.0: open plateau ~1.0, held ~0.03,
    # closed-empty ~0.0 — verified on both YAM datasets). NOT the 0-2.4 action scale.
    open_width: float = 0.85             # gripper_pos above this = open
    closed_width: float = 0.5            # gripper_pos below this = closing/closed


def main(args: Args) -> None:
    ep = pathlib.Path(args.episode_dir)
    states = np.load(ep / f"{args.arm}_states.npz")
    jp = states["joint_pos"]                                   # (N, 6)
    gp = states["gripper_pos"].reshape(len(jp), -1)[:, 0]      # (N,)
    eff = states["joint_eff"] if "joint_eff" in states and len(states["joint_eff"]) else None
    rec_ee = states["ee_pose"] if "ee_pose" in states and len(states["ee_pose"]) else None
    mask = np.ones(len(jp), bool)
    if args.phase and (ep / "phase.npy").exists():
        ph = np.load(ep / "phase.npy")[: len(jp)]
        mask = np.asarray(ph) == args.phase
        logger.info("phase filter '{}': {}/{} frames kept", args.phase, int(mask.sum()), len(jp))
        if mask.sum() < 30:
            raise SystemExit(f"too few '{args.phase}' frames ({int(mask.sum())}) in this episode")
    jp, gp = jp[mask], gp[mask]
    if eff is not None:
        eff = eff[: len(mask)][mask]
    if rec_ee is not None:
        rec_ee = rec_ee[: len(mask)][mask] if len(rec_ee) == len(mask) else None
    logger.info("episode: {} analyzed steps, keys={}", len(jp), list(states.keys()))

    # ---- FK -> ee positions (arm base frame; identical to runtime) ----
    fk = EEPoseInjector(sides=[args.arm])
    poses = np.stack([fk.ee_pose(q) for q in jp])              # (N, 7) [qw qx qy qz x y z]
    xyz = poses[:, 4:7]
    if rec_ee is not None:
        d = float(np.nanmean(np.linalg.norm(rec_ee[:, 4:7] - xyz, axis=1)))
        logger.info("recorded ee_pose present — mean |recorded - FK| = {:.4f} m "
                    "(should be ~0 if the teleop agent filled ee_pose from the same frame)", d)

    # ---- TABLE_Z: the lowest sustained ee height (touch points) ----
    z = xyz[:, 2]
    table_z = float(np.percentile(z, 1.0))
    logger.info("TABLE_Z (1st percentile of ee z): {:.4f} m   [min={:.4f}, median={:.4f}]",
                table_z, float(z.min()), float(np.median(z)))

    # ---- PICKUP_REGION: xy bounds while sweeping near table height ----
    near = z < table_z + args.near_table_band_m
    if near.sum() >= 10:
        xs, ys = xyz[near, 0], xyz[near, 1]
        # 2nd/98th percentiles reject stray dips; shrink by 1 cm for a safety margin
        x0, x1 = np.percentile(xs, [2, 98]) + [0.01, -0.01]
        y0, y1 = np.percentile(ys, [2, 98]) + [0.01, -0.01]
        logger.info("PICKUP_REGION = ({:.3f}, {:.3f}, {:.3f}, {:.3f})  (x_min,x_max,y_min,y_max; "
                    "{} near-table samples)", x0, x1, y0, y1, int(near.sum()))
    else:
        logger.warning("too few near-table samples ({}) — sweep the region closer to the table",
                       int(near.sum()))

    # ---- gripper-effort thresholds (review C6) ----
    if eff is not None:
        if eff.shape[1] >= 7:
            ge = np.abs(eff[:, 6])                              # gripper motor effort
            open_m = gp >= args.open_width
            closed_m = gp <= args.closed_width
            p95_open = float(np.percentile(ge[open_m], 95)) if open_m.any() else float("nan")
            p50_closed = float(np.percentile(ge[closed_m], 50)) if closed_m.any() else float("nan")
            p90_closed = float(np.percentile(ge[closed_m], 90)) if closed_m.any() else float("nan")
            logger.info("gripper effort |eff[6]|: open p95={:.3f}  closed p50={:.3f} p90={:.3f}",
                        p95_open, p50_closed, p90_closed)
            logger.info("suggest LOAD_LOW  ~ {:.3f}  (just above open-gripper noise)",
                        p95_open * 1.5 if np.isfinite(p95_open) else float("nan"))
            logger.info("suggest LOAD_HOLD ~ {:.3f}  (between empty-closed and holding-a-vial — "
                        "verify against the squeeze-and-hold window in the episode)", p50_closed)
        else:
            logger.warning("joint_eff has {} entries (no gripper column) — the driver does not "
                           "report the gripper motor effort; the verifier will fall back to "
                           "width+lift (contact_load sentinel -1)", eff.shape[1])
    else:
        logger.warning("no joint_eff in the episode — re-record with the updated EpisodeRecorder "
                       "(it now saves joint_eff)")

    logger.info("\nApply: configs/yam_subtask_rl_grasp.yaml -> verifiers.table_z + reset PICKUP_REGION; "
                "artifacts verifiers.py -> LOAD_LOW/LOAD_HOLD.")


if __name__ == "__main__":
    main(tyro.cli(Args))
