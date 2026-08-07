"""Closed-loop kinematic test of the EAP inverse reset with the IK-servo primitives (H16).

Runs the FULL Codex-authored VialsGraspReset through ResetControl against a fake
position-controlled arm: commanded joints -> next tick's observed joints; ee_pose via the
same pinocchio FK the runtime uses; gripper ACTION (0-2.4) -> OBSERVED width (0-1).
Proves: non-blocking per-tick actions (bounded joint steps, right arm only), the
carry -> lower -> release -> raise phase machine converges, the vial is released at the
randomized target, and done() fires — with NO robot and NO detector (fail-open paths).

Run:  uv run python limb/agents/policy_learning/subtask/test_reset_ik_servo.py
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
    # gripper motor effort: high while squeezing the held vial (width 0.03), low otherwise
    g_eff = 0.9 if width < 0.5 else 0.05
    return {"right": {"joint_pos": q.copy(), "joint_vel": np.zeros(7), "gripper_pos": [width],
                      "joint_eff": [0.0] * 6 + [g_eff], "ee_pose": fk.ee_pose(q)},
            "left": {"joint_pos": np.zeros(6), "joint_vel": np.zeros(7), "gripper_pos": [0.5],
                     "joint_eff": [0.0] * 7},
            "right_wrist_camera": {"images": {"rgb": IMG}},
            "head_camera": {"images": {"rgb": IMG}},
            "left_wrist_camera": {"images": {"rgb": IMG}}}


class FakeDetector:
    """Vial always visible (as if the image detector found it on the table)."""
    def detect_vial(self, rgb, cam="default"):
        return {"found": True, "area": 60.0, "point": [10.0, 10.0], "bbox": [0, 0, 8, 8]}

    def vial_visible(self, obs, side="right"):
        return True


def run_reset(seed, detector=None, place_z=None, floor=None, xy_lock=None, stiction=None):
    reset = make_reset(table_z=TABLE_Z, control="ik_servo", seed=seed, detector=detector)
    if place_z is not None:
        reset.inner.place_z = place_z     # what the agent plumbs in from verifier.last_grasp_z
    q = np.zeros(6)                       # FK -> ee ~ (0.11, 0.0, 0.164): held above the table
    xy0 = np.asarray(fk.ee_pose(q)[4:6], float)
    grip_cmd = 0.0                        # vial HELD (closed)
    width = 0.03                          # observed held-on-vial width
    release_xy, release_z, max_step, left_cmds = None, None, 0.0, 0
    target = None                          # anchored to the pickup xy on the first act tick
    for t in range(600):
        o = obs_of(q, width)
        if reset.done(o):
            return dict(done_at=t, release_xy=release_xy, release_z=release_z, target=target,
                        max_step=max_step, left_cmds=left_cmds, failed=reset.failed)
        a = reset.act(o)
        if target is None and reset.inner._target_xy is not None:
            target = tuple(reset.inner._target_xy)
        if "left" in a:
            left_cmds += 1
        if "right" in a:
            q_cmd = np.asarray(a["right"]["pos"][:6], float)
            max_step = max(max_step, float(np.max(np.abs(q_cmd - q))))
            new_grip = float(a["right"]["pos"][6])
            if new_grip > 1.1 and grip_cmd <= 1.1:            # release moment
                release_xy = np.asarray(fk.ee_pose(q_cmd)[4:6], float)
                release_z = float(fk.ee_pose(q)[6])
            grip_cmd = new_grip
            # PHYSICAL FLOOR: commands that would take the EE below `floor` are not
            # tracked (hard stop against the table) — models the 2026-07-06 21:24 run
            # where the offline LOWER target was below the arm's reachable minimum.
            # Upward commands are always tracked. XY_LOCK: a SOFT workspace boundary —
            # commands may hold/slide at the boundary (descent still works there, as on
            # a real arm at a joint limit) but not push meaningfully further out.
            # Models the 21:59 carry wedge (far xy unreachable with orientation held).
            ee_cmd = fk.ee_pose(q_cmd)
            z_ok = (floor is None or float(ee_cmd[6]) >= floor
                    or float(ee_cmd[6]) >= float(fk.ee_pose(q)[6]))
            d_cmd = float(np.linalg.norm(np.asarray(ee_cmd[4:6], float) - xy0))
            d_cur = float(np.linalg.norm(np.asarray(fk.ee_pose(q)[4:6], float) - xy0))
            xy_ok = xy_lock is None or d_cmd <= xy_lock or d_cmd <= d_cur + 0.002
            if z_ok and xy_ok:
                if stiction is not None:
                    # STICTION model (2026-07-06 22:24 run): a joint only breaks away
                    # once the PD error (cmd - obs) exceeds the breakaway threshold.
                    # Stepping targets from the OBSERVED joints froze the real arm in
                    # every reset run; command integration must build the lead past this.
                    q = np.where(np.abs(q_cmd - q) >= stiction, q_cmd, q)
                else:
                    q = q_cmd                                  # ideal position control
            width = 1.0 if grip_cmd > 1.1 else 0.03            # open plateau vs held-on-vial
    return dict(done_at=None, release_xy=release_xy, release_z=release_z, target=target,
                max_step=max_step, left_cmds=left_cmds, failed=reset.failed)


def main():
    r1 = run_reset(seed=1)
    check("reset completes (done fires)", r1["done_at"] is not None)
    check("reset not marked failed", not r1["failed"])
    check("gripper released (release recorded)", r1["release_xy"] is not None)
    if r1["release_xy"] is not None:
        d = float(np.linalg.norm(r1["release_xy"] - np.asarray(r1["target"])))
        check(f"vial released AT the random target (d={d:.3f} m)", d < 0.03)
    check(f"per-tick joint step bounded (max={r1['max_step']:.3f} rad)", r1["max_step"] <= 0.021)
    check("left arm never commanded", r1["left_cmds"] == 0)
    tgt_in = (0.179 <= r1["target"][0] <= 0.392) and (-0.214 <= r1["target"][1] <= 0.187)
    check("target sampled inside calibrated PICKUP_REGION", tgt_in)

    r2 = run_reset(seed=2)
    check("second reset completes", r2["done_at"] is not None)
    check("placement RANDOMIZED across resets",
          float(np.linalg.norm(np.asarray(r1["target"]) - np.asarray(r2["target"]))) > 0.01)

    # ---- cold boot: NOTHING held, gripper closed-empty, vial on the table -------
    # The reset must ESTABLISH the start state itself (open gripper + rise), verified
    # by done() with the image detector — no human, no failure (user rule: the loop
    # must be bootable with the REAL verifier/reset; there is no scripted stand-in).
    reset = make_reset(table_z=TABLE_Z, control="ik_servo", seed=3, detector=FakeDetector())
    q = np.zeros(6)
    grip_cmd, width = 0.0, 0.005                    # closed on AIR (empty)
    done_at = None
    for t in range(300):
        o = obs_of(q, width)
        if reset.done(o):
            done_at = t
            break
        a = reset.act(o)
        if "right" in a:
            q_cmd = np.asarray(a["right"]["pos"][:6], float)
            grip_cmd = float(a["right"]["pos"][6])
            q = q_cmd
            width = 1.0 if grip_cmd > 1.1 else 0.005
    check("cold-boot staging completes (no human)", done_at is not None)
    check("cold-boot did NOT mark failed", not reset.failed)
    check("staging OPENED the gripper", width >= 0.85)

    # ---- mid-carry DROP: vial slips (width leaves held band) -> recoverable re-stage
    reset4 = make_reset(table_z=TABLE_Z, control="ik_servo", seed=4, detector=FakeDetector())
    q = np.zeros(6); width = 0.03; grip_cmd = 0.0
    dropped = False; done_at = None
    for t in range(400):
        o = obs_of(q, width)
        if reset4.done(o):
            done_at = t
            break
        a = reset4.act(o)
        if t == 25 and not dropped:          # vial slips mid-carry: width collapses to empty
            width = 0.004; dropped = True
        if "right" in a:
            q = np.asarray(a["right"]["pos"][:6], float)
            grip_cmd = float(a["right"]["pos"][6])
            if not dropped:
                width = 1.0 if grip_cmd > 1.1 else 0.03
            elif grip_cmd > 1.1:
                width = 1.0                   # staging opens the gripper
    check("mid-carry drop -> re-stage completes (no human)", done_at is not None)
    check("mid-carry drop not marked failed", not reset4.failed)

    # ---- UNREACHABLE release height (2026-07-06 21:24 run): the offline LOWER target
    # sits BELOW the arm's physical floor. The un-gated contact stall must release AT
    # the floor (vial pressing the table) instead of grinding out the step budget.
    r5 = run_reset(seed=5, detector=FakeDetector(), floor=0.13)
    check("floor-blocked: reset completes (no timeout/Call-Human)",
          r5["done_at"] is not None and not r5["failed"])
    check(f"floor-blocked: contact-stall released at the floor (z={r5['release_z']})",
          r5["release_z"] is not None and 0.125 <= r5["release_z"] <= 0.16)

    # ---- live grasp-height anchor: place_z (verifier.last_grasp_z, plumbed by the
    # agent on RESET entry) overrides the offline LOWER_Z_OFF as the release height.
    r6 = run_reset(seed=6, detector=FakeDetector(), place_z=0.150)
    check("place_z: reset completes", r6["done_at"] is not None and not r6["failed"])
    check(f"place_z: released at the LIVE grasp height (z={r6['release_z']})",
          r6["release_z"] is not None and abs(r6["release_z"] - 0.155) < 0.015)

    # ---- CARRY WEDGE (2026-07-06 21:59 run): xy motion constrained -> the carry can
    # never reach the sampled target. The stall escape must PLACE HERE (current xy)
    # instead of grinding out the budget and calling the human with the vial held.
    # (xy_lock 0.10 leaves room for the table-facing reorientation + local descent;
    # a fully-locked arm is the
    # everything-unreachable case where parking for the human IS correct.)
    r7 = run_reset(seed=7, detector=FakeDetector(), xy_lock=0.10, floor=0.13)
    check("carry-wedged: reset completes via place-here (no Call-Human)",
          r7["done_at"] is not None and not r7["failed"])
    check("carry-wedged: gripper actually released the vial", r7["release_xy"] is not None)
    check(f"carry-wedged: released LOW, not at carry height (z={r7['release_z']})",
          r7["release_z"] is not None and r7["release_z"] <= 0.18)   # demo band: vial
          # meets the table at ~0.148; wedged release lands within ~2 cm of it,
          # well below the carry height (0.185+)

    # ---- IMAGE-CONFIRMED RELEASE (user rule 2026-07-07): with a release_check
    # backend wired, the reset must hold at the table until the WRIST camera says
    # upright+near_table, then open. The stub denies the first 20 queries.
    class ReleaseCheckDetector(FakeDetector):
        def __init__(self):
            self.calls = 0

        def check_release(self, rgb, cam="right_wrist_camera"):
            self.calls += 1
            ok = self.calls > 20
            return {"upright": ok, "near_table": True}

    det9 = ReleaseCheckDetector()
    r9 = run_reset(seed=11, detector=det9)
    check("image-confirmed release: waits for the wrist camera's upright verdict",
          det9.calls > 20 and r9["done_at"] is not None and not r9["failed"]
          and r9["release_xy"] is not None)

    # ---- STICTION ARM (2026-07-06 22:24 run — the arm never moved in ANY reset):
    # joints break away only when the command leads the observed position by >= 0.03
    # rad. The old observed-relative stepping kept the lead at step_rad (0.01/0.02)
    # forever -> frozen; command integration must complete the full place-back.
    r8 = run_reset(seed=8, detector=FakeDetector(), stiction=0.03)
    check("stiction arm: reset completes (command integration beats breakaway)",
          r8["done_at"] is not None and not r8["failed"])
    check("stiction arm: released at the random target "
          f"(d={None if r8['release_xy'] is None else round(float(np.linalg.norm(r8['release_xy'] - np.asarray(r8['target']))), 3)})",
          r8["release_xy"] is not None
          and float(np.linalg.norm(r8["release_xy"] - np.asarray(r8["target"], float))) < 0.04)

    # ---- FALLEN VIAL (user rule 2026-07-06): visible but lying on its side (squat
    # head-camera bbox) is a DEGRADING failure -> unrecoverable/Call-Human after the
    # hysteresis; an upright (tall/square) bbox never escalates.
    from subtask.vials_artifacts import make_verifiers

    class FlatDetector(FakeDetector):
        def detect_vial(self, rgb, cam="default"):
            return {"found": True, "area": 400.0, "point": [20.0, 5.0],
                    "bbox": [0.0, 0.0, 40.0, 10.0]}          # w=40, h=10: lying on its side

    o = obs_of(np.zeros(6), 1.0)                              # open gripper, arm above table
    vf = make_verifiers(table_z=TABLE_Z, detector=FlatDetector())
    verdicts = [vf.evaluate(o) for _ in range(120)]
    check("fallen vial: instantaneous vial_fallen flag (gates HUMAN resume)",
          verdicts[0]["vial_fallen"])
    # confirmed-fallen termination requires a GRASP ATTEMPT (user rule 2026-07-08:
    # the human guarantees a standing vial at entry, so pre-grasp fallen verdicts
    # are detector mislabels) — with the gripper fully open it must NOT escalate:
    check("fallen vial: NO escalation without a grasp attempt (open gripper)",
          not verdicts[-1]["unrecoverable"])
    # mid-close gripper (width 0.45 <= CLOSE_ATTEMPT_WIDTH) = attempt in progress;
    # sustained fallen must then escalate after the hysteresis:
    o_close = obs_of(np.zeros(6), 0.45)
    vf1b = make_verifiers(table_z=TABLE_Z, detector=FlatDetector())
    verdicts_c = [vf1b.evaluate(o_close) for _ in range(120)]
    check("fallen vial: escalates unrecoverable after hysteresis + grasp attempt",
          verdicts_c[-1]["unrecoverable"] and "FELL OVER" in verdicts_c[-1]["reason"])
    vf2 = make_verifiers(table_z=TABLE_Z, detector=FakeDetector())   # square bbox = upright-ish
    check("upright vial: never escalates fallen",
          not any(v["unrecoverable"] for v in (vf2.evaluate(o_close) for _ in range(120))))

    # ---- WRIST-PRIMARY pose (user rule 2026-07-07): the right wrist close-up is
    # authoritative — wrist says STANDING, the (far, unreliable) head says fallen ->
    # the scene is a valid start; no escalation, vial_standing stays True.
    class WristStandingHeadFallen(FakeDetector):
        def detect_vial(self, rgb, cam="default"):
            if cam == "right_wrist_camera":
                return {"found": True, "area": 400.0, "point": [10.0, 20.0],
                        "bbox": [0.0, 0.0, 10.0, 40.0],
                        "standing_count": 1, "fallen_count": 0}
            return {"found": True, "area": 400.0, "point": [20.0, 5.0],
                    "bbox": [0.0, 0.0, 40.0, 10.0],
                    "standing_count": 0, "fallen_count": 1}

    vf6 = make_verifiers(table_z=TABLE_Z, detector=WristStandingHeadFallen())
    verdicts6 = [vf6.evaluate(o) for _ in range(120)]
    check("wrist-primary: standing wrist verdict overrides fallen head verdict",
          all(v["vial_standing"] for v in verdicts6)
          and not any(v["unrecoverable"] for v in verdicts6))

    # ---- place_z capture band: mid-lift re-grip rising edges must NOT contaminate
    # the release height. BAND WIDENED to table_z+0.19 (=0.255) 2026-08-05 from REAL
    # rollout data: live closes in the CURRENT scene run z 0.194-0.234 and must be
    # ACCEPTED (the old 0.180 rail rejected all of them, degrading place_z AND the
    # success rule's lift base to stale fallbacks); the observed mid-lift re-grip
    # (z 0.346) stays rejected.
    def at_z(width, zz):
        o2 = obs_of(np.zeros(6), width)
        ee = np.array(o2["right"]["ee_pose"], float)
        ee[6] = zz
        o2["right"]["ee_pose"] = ee
        return o2

    vf5 = make_verifiers(table_z=TABLE_Z, detector=FakeDetector())
    vf5.evaluate(at_z(0.047, 0.346))                   # bogus HIGH re-grip (real 0729 case)
    check("place_z: high (mid-lift) rising edge ignored", vf5.last_grasp_z is None)
    vf5.evaluate(at_z(1.0, 0.346))                     # open -> clears holding edge
    vf5.evaluate(at_z(0.047, 0.234))                   # REAL current-scene table close
    check("place_z: real current-scene close (0.234) captured",
          vf5.last_grasp_z is not None and abs(vf5.last_grasp_z - 0.234) < 1e-9)
    vf5.evaluate(at_z(1.0, 0.25))                      # open -> clears holding edge
    vf5.evaluate(at_z(0.047, 0.11))                    # low-band grasp still captured
    check("place_z: table-height grasp captured",
          vf5.last_grasp_z is not None and abs(vf5.last_grasp_z - 0.11) < 1e-9)
    vf5.evaluate(at_z(1.0, 0.20))
    vf5.evaluate(at_z(0.047, 0.30))                    # later high re-grip
    check("place_z: later high re-grip keeps the table value",
          abs(vf5.last_grasp_z - 0.11) < 1e-9)

    # ---- HOLDING gates vision (user rule 2026-07-06 22:35): a vial INSIDE the gripper
    # is occluded from every camera — vial-lost must never escalate while proprio says
    # holding; with the gripper open and the vial truly gone it still must escalate.
    class NotFoundDetector(FakeDetector):
        def detect_vial(self, rgb, cam="default"):
            return {"found": False, "area": 0.0, "point": None, "bbox": None}

    o_hold = obs_of(np.zeros(6), 0.047)                   # held-on-vial width + load
    vf3 = make_verifiers(table_z=TABLE_Z, detector=NotFoundDetector())
    check("holding vial: all-cameras-missing NEVER escalates",
          not any(vf3.evaluate(o_hold)["unrecoverable"] for _ in range(200)))
    # lost counting requires the wrist to be VIEWING the table (EE over the region):
    # place the test EE inside it; the park pose (x~0.10) must NOT count (see below).
    o_open = obs_of(np.zeros(6), 1.0)                     # open, vial truly gone
    ee_v = np.array(o_open["right"]["ee_pose"], float); ee_v[4] = 0.30
    o_open["right"]["ee_pose"] = ee_v
    vf4 = make_verifiers(table_z=TABLE_Z, detector=NotFoundDetector())
    check("open + vial gone (wrist over the table): escalates lost after hysteresis",
          any(vf4.evaluate(o_open)["unrecoverable"] for _ in range(200)))
    o_park = obs_of(np.zeros(6), 1.0)                     # park pose x~0.10: NOT viewing
    vf4b = make_verifiers(table_z=TABLE_Z, detector=NotFoundDetector())
    check("open + nothing seen from the PARK pose: never escalates (no evidence)",
          not any(vf4b.evaluate(o_park)["unrecoverable"] for _ in range(200)))
    # the reset's own _unrecoverable must behave the same while holding
    reset9 = make_reset(table_z=TABLE_Z, control="ik_servo", seed=9, detector=NotFoundDetector())
    ok9 = True
    for _ in range(200):
        reset9.act(o_hold)
        ok9 = ok9 and not reset9.failed
    check("reset holding vial: cameras-blind never marks failed", ok9)

    # ---- IMAGE-CONFIRMED SUCCESS (user rule 2026-07-07): the verifier's grasp+lift
    # success also needs the wrist camera to see the vial CLENCHED in the fingers.
    class HeldDetector(FakeDetector):
        def __init__(self, held):
            self.held = held

        def check_held(self, rgb, cam="right_wrist_camera"):
            return {"held": self.held}

    def run_success(det):
        vf = make_verifiers(table_z=TABLE_Z, detector=det, stable_frames=15)
        vf.evaluate(at_z(0.047, 0.15))                 # grasp at table -> last_grasp_z
        return any(vf.evaluate(at_z(0.047, 0.25))["success"] for _ in range(40))

    check("success: proprio hold + wrist image CONFIRMS held -> success",
          run_success(HeldDetector(held=True)))
    check("success: proprio hold but wrist image DENIES held -> blocked",
          not run_success(HeldDetector(held=False)))

    # ---- PROPRIO-ONLY success option (user 2026-07-07: the original zero-latency
    # method — the held vial is invisible to cameras by design): with
    # success_image_confirm=False the image verdict is IGNORED entirely.
    vf7 = make_verifiers(table_z=TABLE_Z, detector=HeldDetector(held=False),
                         stable_frames=15, success_image_confirm=False)
    vf7.evaluate(at_z(0.047, 0.15))
    check("success: proprio-only option ignores the image deny (zero-delay mode)",
          any(vf7.evaluate(at_z(0.047, 0.25))["success"] for _ in range(40)))

    # ---- SAM3 geometric pose rule (boxes -> standing/fallen, no VLM judgment) ------
    from subtask.perception import VialDetector
    dget = VialDetector(backend="sam3", throttle_s=999)
    r_stand = dget._boxes_to_pose_result([[100, 100, 130, 128]])          # compact cap blob
    r_fall = dget._boxes_to_pose_result([[100, 100, 190, 126]])          # elongated tube
    r_mix = dget._boxes_to_pose_result([[100, 100, 190, 126], [10, 10, 38, 40]])
    r_none = dget._boxes_to_pose_result([])
    check("sam3 pose: compact box = standing",
          r_stand["found"] and r_stand["standing_count"] == 1 and r_stand["fallen_count"] == 0)
    check("sam3 pose: elongated box = fallen",
          r_fall["found"] and r_fall["fallen_count"] == 1 and r_fall["standing_count"] == 0)
    check("sam3 pose: mixed scene counts both, primary = the STANDING vial",
          r_mix["standing_count"] == 1 and r_mix["fallen_count"] == 1
          and r_mix["pose"] == "standing" and r_mix["bbox"][0] == 10.0)
    check("sam3 pose: empty = not found", not r_none["found"] and r_none["count"] == 0)

    print("\n" + ("ALL PASS" if all(PASS) else f"{PASS.count(False)} FAILURES"))
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
