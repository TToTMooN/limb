"""Refine the CODE-AS-POLICY auto-reset from human grasp-then-put-back demos.

The reset stays authored code (no SFT / learned inverse policy — user decision
2026-07-06); the demos are CALIBRATION+REFERENCE data for its constants:

  record (~20-50 episodes, normal teleop recording):
      vial standing on the table -> grasp -> lift ~10 cm -> carry to a DIFFERENT
      random spot in the pickup region -> lower until the vial TOUCHES the table
      -> open -> retract up. Vary pick + place positions.
      uv run limb record   # or your usual record config

  analyze:
      uv run python scripts/data/subrl_reset_calibrate.py \
          --episodes-dir recordings/<task>          # every episode_* inside
      uv run python scripts/data/subrl_reset_calibrate.py \
          --episodes-dir recordings/<task>/episode_20260706_...   # single episode

Per place-back event (gripper width open->held = GRASP, held->open = RELEASE) it
measures, all via the SAME pinocchio FK frame the runtime reset servos in:
  - grasp_z vs release_z            -> validates the place_z (live grasp height) anchor
  - release_z - table_z             -> the LOWER fallback offset
  - carry peak - table_z            -> CARRY_Z_OFF
  - wrist rotation grasp->release   -> the ori_error release gate (currently 0.35 rad)
  - final-approach descent speed    -> IK step_rad / stall threshold sanity
  - hover-at-release-height frames  -> contact-stall tick count sanity
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
    label: str = ""                    # "failure"/"success": keep only episodes with that
                                       # marker file (put-back demos are labeled FAILURE in
                                       # the 2026-07-06 mixed session; grasps are SUCCESS)
    # OBSERVATION units (0-1.0 measured width: open plateau ~1.0, held-on-vial ~0.05)
    open_width: float = 0.85
    held_min: float = 0.015
    held_max: float = 0.15
    confirm_frames: int = 8            # frames a new width state must persist (debounce)
    table_z: float = 0.065             # runtime TABLE_Z, for offset suggestions
    fps: float = 30.0


def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Rotation angle (rad) between two [qw qx qy qz] quaternions."""
    d = abs(float(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2) + 1e-12))
    return 2.0 * float(np.arccos(min(1.0, d)))


def _events(gp: np.ndarray, a: Args) -> list[tuple[int, int]]:
    """(grasp_idx, release_idx) pairs from the width timeline, debounced.

    A closing gripper passes THROUGH the mid band (1.0 -> ~0.5 -> 0.07), so the
    debounced transition into 'held' arrives from 'mid', not 'open' — grasp = any
    entry into 'held' after the gripper has been open; release = entry into 'open'
    while a grasp is active."""
    def state(w):
        if w >= a.open_width:
            return "open"
        if a.held_min <= w <= a.held_max:
            return "held"
        return "mid"

    events, cur, cand, cand_n = [], state(gp[0]), None, 0
    # Put-back demos START with the vial already held (the reset's precondition) —
    # frame 0 is then the hold start. Release = width leaving the held band UPWARD
    # (humans release to a partial open, 0.4-0.7, not the full 0.85+ plateau).
    grasp_i = 0 if state(gp[0]) == "held" else None
    for i in range(1, len(gp)):
        s = state(gp[i])
        if s == cur:
            cand, cand_n = None, 0
            continue
        if s != cand:
            cand, cand_n = s, 1
        else:
            cand_n += 1
        if cand_n >= a.confirm_frames:
            cur = cand
            cand, cand_n = None, 0
            j = i - a.confirm_frames + 1              # transition onset
            if cur in ("open", "mid") and grasp_i is not None and gp[j] > a.held_max:
                events.append((grasp_i, j))           # released (width left band upward)
                grasp_i = None
            elif cur == "held" and grasp_i is None:
                grasp_i = j
    return events


def main(args: Args) -> None:
    root = pathlib.Path(args.episodes_dir)
    eps = sorted(root.glob("episode_*")) if not (root / f"{args.arm}_states.npz").exists() else [root]
    if args.label:
        eps = [e for e in eps if (e / args.label.upper()).exists()]
        logger.info("label filter '{}': {} episode(s) kept", args.label, len(eps))
    if not eps:
        raise SystemExit(f"no episode_* under {root}")
    fk = EEPoseInjector(sides=[args.arm])

    rows = []                                          # one row per place-back event
    for ep in eps:
        try:
            states = np.load(ep / f"{args.arm}_states.npz")
            jp = states["joint_pos"]
            gp = states["gripper_pos"].reshape(len(jp), -1)[:, 0]
        except Exception as e:
            logger.warning("{}: skipped ({})", ep.name, e)
            continue
        poses = np.stack([fk.ee_pose(q) for q in jp])  # (N,7) [qw qx qy qz x y z]
        z, quat = poses[:, 6], poses[:, 0:4]
        for g, r in _events(gp, args):
            if r - g < 15:                             # spurious blip
                continue
            seg_z = z[g:r + 1]
            peak_i = g + int(np.argmax(seg_z))
            # descent start: last frame within 1 cm of the carry peak before release
            near_peak = np.nonzero(seg_z >= seg_z.max() - 0.01)[0]
            desc_start = g + int(near_peak[-1])
            last = slice(max(g, r - 15), r + 1)        # final ~0.5 s of approach
            appr_speed = float(-np.mean(np.diff(z[last]))) * args.fps if r - g > 15 else float("nan")
            hover = int(np.sum(np.abs(z[max(g, r - 90):r] - z[r]) < 0.004))
            rows.append(dict(
                ep=ep.name, grasp_z=float(z[g]), release_z=float(z[r]),
                dz_place=float(z[r] - z[g]), carry_off=float(seg_z.max() - args.table_z),
                ori_rad=_quat_angle(quat[g], quat[r]),
                desc_frames=int(r - desc_start), appr_mps=appr_speed, hover_frames=hover,
                place_xy=(float(poses[r, 4]), float(poses[r, 5])),
            ))

    if not rows:
        raise SystemExit("no grasp->release events found — check widths/units or record longer holds")

    logger.info("{} place-back events from {} episode(s)", len(rows), len(eps))
    for r in rows:
        logger.info("  {}: grasp_z={:.3f} release_z={:.3f} (dz={:+.3f})  carry+{:.3f}  "
                    "ori={:.2f}rad  descent={}f  appr={:.3f}m/s  hover={}f  xy=({:.3f},{:.3f})",
                    r["ep"], r["grasp_z"], r["release_z"], r["dz_place"], r["carry_off"],
                    r["ori_rad"], r["desc_frames"], r["appr_mps"], r["hover_frames"],
                    *r["place_xy"])

    def pct(k, q):
        return float(np.nanpercentile([r[k] for r in rows], q))

    logger.info("\n---- suggested reset constants (artifacts/vials_grasp/reset_policy.py) ----")
    logger.info("release_z p5/p50/p95 = {:.3f}/{:.3f}/{:.3f}  ->  LOWER fallback offset "
                "LOWER_Z_OFF ~ {:.3f} (p50 - table_z {:.3f})",
                pct("release_z", 5), pct("release_z", 50), pct("release_z", 95),
                pct("release_z", 50) - args.table_z, args.table_z)
    logger.info("release_z - grasp_z p50 = {:+.3f} m  (|.|<~5 mm CONFIRMS the place_z="
                "live-grasp-height anchor; larger = humans release higher/lower than the pick)",
                pct("dz_place", 50))
    logger.info("carry peak offset p50 = {:.3f}  ->  CARRY_Z_OFF ~ {:.3f} (current 0.12)",
                pct("carry_off", 50), pct("carry_off", 50))
    logger.info("wrist rotation grasp->release p95 = {:.2f} rad  ->  ori_error release gate "
                "~ {:.2f} (current 0.35)", pct("ori_rad", 95), pct("ori_rad", 95) + 0.1)
    logger.info("final-approach speed p50 = {:.3f} m/s; hover-before-open p50 = {:.0f} frames "
                "(stall detector waits 15) — descent slower than ~2 mm/frame near touch is "
                "NORMAL human behavior; keep the stall window >= that hover",
                pct("appr_mps", 50), pct("hover_frames", 50))
    xs = [r["place_xy"][0] for r in rows]; ys = [r["place_xy"][1] for r in rows]
    logger.info("place xy span: x [{:.3f},{:.3f}]  y [{:.3f},{:.3f}]  (compare PICKUP_REGION)",
                min(xs), max(xs), min(ys), max(ys))


if __name__ == "__main__":
    main(tyro.cli(Args))
