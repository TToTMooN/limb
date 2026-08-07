"""Calibrate SubRL vials-grasp geometry from a LeRobot-converted expert-demo dataset
(e.g. Sichang0621/vials_4_30fps_180 — 180 GELLO teleop demos, `limb convert-lerobot`).

What the demos give us OFFLINE (no robot session):
  - TABLE_Z + grasp heights : FK (pinocchio, same frame as runtime) on right joint_pos
  - PICKUP_REGION           : xy of the RIGHT gripper at grasp onset across all episodes
                              (= the true vial spawn distribution)
  - lift_m                  : how high demos actually lift after grasping
  - gripper width thresholds: OPEN_WIDTH / HELD_WIDTH_MIN/MAX / EMPTY_WIDTH from the
                              width distribution around grasp events
What they CANNOT give: gripper-EFFORT thresholds (joint_eff wasn't recorded pre-fix) and
the FK-vs-driver ee_pose cross-check — those stay in the on-robot Stage 1/2.

Usage:
  uv run python scripts/data/subrl_calibrate_lerobot.py --dataset-dir <path-with-data,meta>
"""

from __future__ import annotations

import json
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
    dataset_dir: str
    right_joints: tuple = (7, 13)      # state[7:13] per the converter's name layout
    right_gripper: int = 13
    fk_stride: int = 2                 # FK every Nth frame (30 fps -> 15 Hz is plenty)
    hold_s: float = 0.5                # closed at least this long after onset = a real grasp


def main(args: Args) -> None:
    base = pathlib.Path(args.dataset_dir)
    import pyarrow.parquet as pq

    info = json.loads((base / "meta/info.json").read_text())
    fps = float(info.get("fps", 30))
    names = info["features"]["observation.state"].get("names") or []
    if names and (names[args.right_joints[0]] != "right_joint_0" or names[args.right_gripper] != "right_gripper"):
        raise SystemExit(f"state layout mismatch: {names}")

    files = sorted(base.glob("data/chunk-*/file-*.parquet"))
    logger.info("{} parquet files, fps={}", len(files), fps)

    fk = EEPoseInjector(sides=["right"])
    all_z, all_w = [], []
    grasp_xy, grasp_z, grasp_w, lift_delta = [], [], [], []
    open_w_plateau = []

    hold_n = max(1, int(args.hold_s * fps))
    for f in files:
        t = pq.read_table(f, columns=["observation.state", "episode_index"])
        S = np.asarray(t["observation.state"].to_pylist(), np.float32)
        for ep in np.unique(np.asarray(t["episode_index"])):
            m = np.asarray(t["episode_index"]) == ep
            s = S[m]
            jr = s[:, args.right_joints[0]:args.right_joints[1]]
            w = s[:, args.right_gripper]
            all_w.append(w)
            # adaptive open/closed split from THIS dataset's width range
            w_lo, w_hi = np.percentile(w, [5, 95])
            thr = 0.5 * (w_lo + w_hi)
            closed = w < thr
            # grasp onsets: open -> closed staying closed >= hold_s
            on = np.flatnonzero(~closed[:-1] & closed[1:]) + 1
            xyz = None
            for k in on:
                if k + hold_n >= len(w) or not closed[k:k + hold_n].all():
                    continue
                if xyz is None:                     # FK lazily, strided, only when needed
                    xyz = np.full((len(jr), 3), np.nan, np.float32)
                    for i in range(0, len(jr), args.fk_stride):
                        p = fk.ee_pose(jr[i])
                        if p is not None:
                            xyz[i] = p[4:7]
                    # forward-fill strided gaps
                    for i in range(1, len(xyz)):
                        if np.isnan(xyz[i, 0]):
                            xyz[i] = xyz[i - 1]
                    all_z.append(xyz[:, 2].copy())
                gz = float(xyz[k, 2])
                grasp_xy.append(xyz[k, :2].copy()); grasp_z.append(gz); grasp_w.append(float(np.median(w[k:k + hold_n])))
                # lift: max height while it stays closed after the grasp
                end = k
                while end < len(w) and closed[end]:
                    end += 1
                lift_delta.append(float(np.nanmax(xyz[k:end, 2]) - gz))
            open_w_plateau.append(np.percentile(w[~closed], 90) if (~closed).any() else np.nan)

    all_wc = np.concatenate(all_w)
    logger.info("gripper width overall: min={:.3f} p5={:.3f} p50={:.3f} p95={:.3f} max={:.3f}",
                *(float(np.percentile(all_wc, q)) for q in (0, 5, 50, 95, 100)))
    if not grasp_z:
        raise SystemExit("no grasp events detected — check the gripper convention/threshold")

    gxy = np.stack(grasp_xy); gz = np.asarray(grasp_z); gw = np.asarray(grasp_w)
    zc = np.concatenate(all_z); ld = np.asarray(lift_delta)
    logger.info("grasp events: {} across {} files", len(gz), len(files))
    logger.info("ee z: overall p1={:.4f} | grasp-onset z: p5={:.4f} p50={:.4f} p95={:.4f}",
                float(np.percentile(zc, 1)), *(float(np.percentile(gz, q)) for q in (5, 50, 95)))
    logger.info("lift after grasp: p25={:.3f} p50={:.3f} p75={:.3f} m",
                *(float(np.percentile(ld, q)) for q in (25, 50, 75)))
    x0, x1 = np.percentile(gxy[:, 0], [2, 98]); y0, y1 = np.percentile(gxy[:, 1], [2, 98])
    ow = float(np.nanmedian(open_w_plateau))

    print("\n===== RECOMMENDED CALIBRATION (from {} expert grasps) =====".format(len(gz)))
    print(f"TABLE_Z        = {float(np.percentile(zc, 1)):.4f}   # p1 of ee z (verify on-robot: grasp z p50 - vial half-height)")
    print(f"PICKUP_REGION  = ({x0 + 0.01:.3f}, {x1 - 0.01:.3f}, {y0 + 0.01:.3f}, {y1 - 0.01:.3f})  # right-arm base frame")
    print(f"lift_m         = {max(0.03, float(np.percentile(ld, 25)) * 0.5):.3f}   # half the demos' p25 lift")
    print(f"OPEN_WIDTH     = {ow * 0.9:.3f}   # 90% of the open plateau ({ow:.3f})")
    print(f"HELD_WIDTH     = [{float(np.percentile(gw, 2)) * 0.8:.3f}, {float(np.percentile(gw, 98)) * 1.2:.3f}]"
          f"   # held-on-vial widths p2..p98 ± margin (raw p50={float(np.percentile(gw, 50)):.3f})")
    print("NOTE: grasp-onset z p50 = {:.4f} — the verifier's 'lifted' is z >= TABLE_Z + lift_m; "
          "check TABLE_Z + lift_m sits between grasp z and transport z.".format(float(np.percentile(gz, 50))))


if __name__ == "__main__":
    main(tyro.cli(Args))
