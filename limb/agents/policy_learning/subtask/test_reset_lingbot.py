"""Offline tests for the lingbot auto-reset upgrades (user 2026-08-05).

Covers the three new behaviors of VialsGraspReset on the fake IK-servo arm
(same harness as test_reset_ik_servo.py):
  1. RELEASE HEIGHT GATE — prot_m above NEAR_TABLE_PROT_M blocks the gripper
     from opening; the release happens only once the metric height confirms the
     vial bottom is at the table. SAM3-style verdicts (no detections/prot_m)
     fall back to the old proprio path.
  2. FALLEN WATCH — a sustained fallen verdict with nothing held freezes the
     arm (hold actions) and surfaces failed=True for the Call-Human escalation;
     a fallen verdict while the vial is HELD is ignored (occluded close-up).
  3. RAISE ABOVE THE WINDOW — the post-release retract ends above the
     selector's approach window (z_off 0.19 > approach_z_max 0.18).

Run:  uv run python limb/agents/policy_learning/subtask/test_reset_lingbot.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from subtask.fk import EEPoseInjector
from subtask.vials_artifacts import make_reset

PASS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    PASS.append(bool(cond))


TABLE_Z = 0.065
fk = EEPoseInjector(sides=["right"])
IMG = np.zeros((8, 8, 3), np.uint8)


def obs_of(q, width):
    g_eff = 0.9 if width < 0.5 else 0.05
    return {"right": {"joint_pos": q.copy(), "joint_vel": np.zeros(7), "gripper_pos": [width],
                      "joint_eff": [0.0] * 6 + [g_eff], "ee_pose": fk.ee_pose(q)},
            "left": {"joint_pos": np.zeros(6), "joint_vel": np.zeros(7), "gripper_pos": [0.5],
                     "joint_eff": [0.0] * 7},
            "right_wrist_camera": {"images": {"rgb": IMG}},
            "head_camera": {"images": {"rgb": IMG}},
            "left_wrist_camera": {"images": {"rgb": IMG}}}


class LingbotDet:
    """Fake lingbot_depth verdicts: metric prot_m + fallen counts, mutable mid-run."""

    def __init__(self, prot=0.16, fallen=0, found=True, with_detections=True):
        self.prot = prot
        self.fallen = fallen
        self.found = found
        self.with_detections = with_detections

    def detect_vial(self, rgb, cam="default"):
        dets = ([{"prot_m": self.prot, "pose": "standing"}]
                if self.found and self.with_detections else [])
        return {"found": self.found, "standing_count": 1 if self.found else 0,
                "fallen_count": self.fallen, "pose": "standing", "detections": dets}

    def vial_visible(self, obs, side="right"):
        return True


class DynamicDet(LingbotDet):
    """prot_m tracks the ARM: the vial hangs from the gripper, so its measured
    protrusion = own height (0.075) + how far the gripper still is above the true
    per-rollout release height. Models the user's point: the grasp point differs
    every episode, so only the live measurement knows when the bottom is at the
    table."""

    def __init__(self, true_release_z):
        super().__init__()
        self.true_release_z = true_release_z
        self.z = 1.0

    def detect_vial(self, rgb, cam="default"):
        self.prot = 0.075 + max(0.0, self.z - self.true_release_z)
        return super().detect_vial(rgb, cam)


def run_place(det, flip_at=None, flip_prot=0.08, ticks=900):
    """Held-vial place-back on the ideal fake arm; returns release/done telemetry."""
    reset = make_reset(table_z=TABLE_Z, control="ik_servo", seed=7, detector=det)
    reset.inner.place_z = getattr(det, "place_z_override", TABLE_Z + 0.083)
    q = np.zeros(6)
    grip_cmd, width = 0.0, 0.03
    release_at, done_at, done_z, release_z = None, None, None, None
    for t in range(ticks):
        if flip_at is not None and t == flip_at:
            det.prot = flip_prot
        if hasattr(det, "z"):
            det.z = float(fk.ee_pose(q)[6])
        o = obs_of(q, width)
        if reset.done(o):
            done_at, done_z = t, float(fk.ee_pose(q)[6])
            break
        a = reset.act(o)
        if "right" in a:
            q_cmd = np.asarray(a["right"]["pos"][:6], float)
            new_grip = float(a["right"]["pos"][6])
            if new_grip > 1.1 and grip_cmd <= 1.1 and release_at is None:
                release_at = t
                release_z = float(fk.ee_pose(q)[6])
            grip_cmd = new_grip
            q = q_cmd
            width = 1.0 if grip_cmd > 1.1 else 0.03
    return dict(reset=reset, release_at=release_at, done_at=done_at, done_z=done_z,
                release_z=release_z)


def main():
    # ---- 1a. height gate BLOCKS while the vial reads high ----------------------
    det = LingbotDet(prot=0.16)
    r = run_place(det, flip_at=450, ticks=900)
    blocked = r["release_at"] is None or r["release_at"] >= 450
    check("release blocked while prot_m 0.16 > gate 0.10", blocked)
    check("release fires after prot_m drops to 0.08", r["release_at"] is not None)
    check("reset completes after gated release", r["done_at"] is not None)
    check("no failure during the gated wait", not r["reset"].failed)

    # ---- 1b. SAM3-style verdict (no prot_m) falls back to proprio release ------
    r2 = run_place(LingbotDet(with_detections=False), ticks=600)
    check("no-prot verdict falls back to proprio release", r2["release_at"] is not None
          and r2["done_at"] is not None)

    # ---- 3. raise ends ABOVE the approach window (0.18) ------------------------
    if r2["done_z"] is not None:
        z_off = r2["done_z"] - TABLE_Z
        check(f"retract above the window (z_off={z_off:.3f} > 0.18)", z_off > 0.18)
    else:
        check("retract above the window", False)

    # ---- 1c. PER-ROLLOUT height (user 2026-08-05): a too-HIGH waypoint (the vial
    # was grasped low on its body this episode) must be corrected by the LIVE
    # measurement — the descent continues BELOW place_z until prot_m says the
    # bottom is at the table; no fixed height decides the release.
    ddet = DynamicDet(true_release_z=TABLE_Z + 0.083)
    ddet.place_z_override = TABLE_Z + 0.14          # wrong waypoint, 5.7 cm too high
    r3 = run_place(ddet, ticks=900)
    check("wrong-high waypoint still releases", r3["release_at"] is not None)
    if r3["release_z"] is not None:
        check(f"release DESCENDED BELOW the waypoint on measurement "
              f"(z={r3['release_z']:.3f} < waypoint {TABLE_Z + 0.14:.3f})",
              r3["release_z"] < TABLE_Z + 0.14 - 0.01)
        check(f"release near the TRUE per-rollout height (z={r3['release_z']:.3f})",
              abs(r3["release_z"] - (TABLE_Z + 0.083)) < 0.03)
    else:
        check("release z recorded", False)
    check("wrong-waypoint episode completes", r3["done_at"] is not None)

    # ---- 2a. fallen watch: empty gripper + sustained fallen -> freeze + failed --
    det_f = LingbotDet(fallen=1)
    reset = make_reset(table_z=TABLE_Z, control="ik_servo", seed=8, detector=det_f)
    q = np.zeros(6)
    frozen_cmds = 0
    for t in range(200):
        o = obs_of(q, 1.0)                      # gripper OPEN — nothing held
        a = reset.act(o)
        if reset.failed and "right" not in a:
            frozen_cmds += 1
    check("sustained fallen (empty gripper) -> failed", reset.failed)
    check("fail_reason names the fallen watch",
          "fallen" in str(getattr(reset.inner, "fail_reason", ""))
          or "fallen" in str(getattr(reset, "fail_reason", "")))
    check("arm FROZEN after fallen escalation (hold, no commands)", frozen_cmds > 0)

    # ---- 2b. fallen verdict while HELD is ignored ------------------------------
    det_h = LingbotDet(fallen=1)
    reset_h = make_reset(table_z=TABLE_Z, control="ik_servo", seed=9, detector=det_h)
    reset_h.inner.place_z = TABLE_Z + 0.083
    q = np.zeros(6)
    for t in range(200):
        reset_h.act(obs_of(q, 0.03))            # vial HELD the whole time
    check("fallen while HELD does not trigger the watch",
          getattr(reset_h.inner, "fail_reason", None) != "fallen vial (lingbot watch)")

    # ---- 2c. done() is blocked while a fallen verdict is active ----------------
    reset_d = make_reset(table_z=TABLE_Z, control="ik_servo", seed=10, detector=LingbotDet())
    q = np.zeros(6)
    o_open = obs_of(q, 1.0)
    reset_d.inner._fallen_frames = 5
    d_blocked = reset_d.done(o_open)
    reset_d.inner._fallen_frames = 0
    check("done() blocked while fallen frames active", not d_blocked)

    print()
    print("ALL PASS" if all(PASS) else f"{PASS.count(False)} FAILURES")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
