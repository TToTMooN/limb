"""Offline tests for the vial-INSERTION sub-task artifacts (RL2, 2026-07-31).

Run: uv run python limb/agents/policy_learning/subtask/test_insert_offline.py
"""

import numpy as np

from limb.agents.policy_learning.subtask.vials_artifacts import make_selector, make_verifiers

INSERT_DIR = "/home/ssc/Desktop/research/SubRL-VLA/real_robot/artifacts/vials_insert"


class StubDetector:
    """detect_vial -> None (fail-open resume cue); check_inserted scripted —
    either a single dict/None for every call, or a LIST consumed one per call."""

    def __init__(self, insert_result=None, stand_visible=None):
        self.insert_result = insert_result
        self.insert_calls = 0
        self.stand_visible = stand_visible
        self.vial_visible = None

    def detect_vial(self, rgb, cam=None):
        return None

    def check_inserted(self, before, after, cam=None, epoch=0):
        self.insert_calls += 1
        if isinstance(self.insert_result, list):
            return self.insert_result.pop(0) if self.insert_result else None
        return self.insert_result

    def check_stand(self, rgb, cam=None):
        if self.stand_visible is None:
            return None
        return {"stand_visible": bool(self.stand_visible)}


def obs(side="right", width=0.03, effort=0.31, xyz=(0.33, 0.25, 0.32)):
    rgb = np.zeros((48, 64, 3), np.uint8)
    arm = {"joint_pos": np.zeros(6, np.float32),
           "gripper_pos": np.array([width], np.float32),
           "joint_eff": np.full(7, effort, np.float32),
           "ee_pose": np.array([1, 0, 0, 0, *xyz], np.float32)}
    other = {"joint_pos": np.zeros(6, np.float32),
             "gripper_pos": np.array([1.0], np.float32),
             "joint_eff": np.zeros(7, np.float32),
             "ee_pose": np.array([1, 0, 0, 0, 0.2, -0.2, 0.3], np.float32)}
    return {side: arm, ("left" if side == "right" else "right"): other,
            "head_camera": {"images": {"rgb": rgb}},
            "left_wrist_camera": {"images": {"rgb": rgb}},
            "right_wrist_camera": {"images": {"rgb": rgb}}}


def mk_verifier(insert_result, **kw):
    kw.setdefault("confirm_timeout_s", 0.2)
    kw.setdefault("confirm_retries", 1)
    return make_verifiers(table_z=0.065, artifacts_dir=INSERT_DIR,
                          class_name="VialsInsertVerifiers", side="right",
                          detector=StubDetector(insert_result),
                          stand_region=(0.287, 0.370, 0.191, 0.319),
                          release_z_band=(0.205, 0.270),
                          min_hold_frames=5, open_ramp_ticks=6,
                          after_delay_frames=3, drop_frames=6, **kw)


def run_episode(v, verdicts_after_release=30, xy=(0.33, 0.25)):
    """hold in region -> REAL open ramp (2 dead-zone ticks, replay bug #1) ->
    open at the stand -> confirm ticks; return last result."""
    for _ in range(8):                                     # sustained hold, in region
        r = v.evaluate(obs(width=0.03, xyz=(*xy, 0.24)))
        assert not r["success"] and not r["unrecoverable"]
    for w in (0.16, 0.35):                                 # the physical open ramp
        r = v.evaluate(obs(width=w, effort=0.02, xyz=(*xy, 0.25)))
        assert not r["success"] and not r["unrecoverable"]
    last = None
    for _ in range(verdicts_after_release):                # gripper open at the stand
        last = v.evaluate(obs(width=0.95, xyz=(*xy, 0.27)))
        if last["success"] or last["unrecoverable"] or last.get("failure"):
            break
    return last


def main():
    # ---- selector ----------------------------------------------------------
    s = make_selector(table_z=0.065, artifacts_dir=INSERT_DIR,
                      class_name="VialsInsertSelector", side="right", start_frames=3,
                      stand_region=(0.287, 0.370, 0.191, 0.319),
                      approach_z_min=0.25, approach_z_max=0.42)
    s.select(obs(), "rl")                                  # mode reset
    assert all(s.select(obs(width=0.03, xyz=(0.33, 0.25, 0.32)), "vla") != "rl"
               for _ in range(2))
    assert s.select(obs(width=0.03, xyz=(0.33, 0.25, 0.32)), "vla") == "rl"
    print("PASS selector fires holding-above-stand after start_frames")

    s.select(obs(), "rl")
    assert all(s.select(obs(width=0.8, xyz=(0.33, 0.25, 0.32)), "vla") is None
               for _ in range(5)), "open gripper must not fire"
    assert all(s.select(obs(width=0.03, xyz=(0.33, -0.10, 0.32)), "vla") is None
               for _ in range(5)), "outside stand region must not fire"
    assert all(s.select(obs(width=0.03, xyz=(0.33, 0.25, 0.22)), "vla") is None
               for _ in range(5)), "release-band z must not fire (too low)"
    assert all(s.select(obs(width=0.03, xyz=(0.33, 0.25, 0.32)), "rl") is None
               for _ in range(5)), "selector proposes nothing in rl"
    print("PASS selector blocks: open gripper / off-region / low z / rl mode")

    # ---- verifier: pair-confirmed outcomes (through the REAL open ramp) ----
    v = mk_verifier({"result": "success"})
    r = run_episode(v)
    assert r["success"] and "pair-confirmed" in r["reason"]
    print("PASS verifier success through the gripper-open ramp (replay bug #1)")

    # ONE-SHOT re-arm (replay bug #2): episode 2 in the SAME scene, no _reset()
    r = v.evaluate(obs(width=0.95, xyz=(0.33, 0.25, 0.27)))
    assert not r["success"], "success must not persist into the next episode"
    r = run_episode(v)
    assert r["success"], "verifier must re-arm for episode 2 without _reset()"
    print("PASS verifier one-shot: auto re-arms for the next episode (replay bug #2)")

    # not_done RETRY with a later after frame (replay bug #4)
    r = run_episode(mk_verifier([{"result": "not_done"}, {"result": "success"}]))
    assert r["success"] and "pair-confirmed" in r["reason"]
    print("PASS verifier retries a later after frame on not_done (replay bug #4)")

    # TWO-WRIST combine: failed on either camera beats success
    d = StubDetector(None)
    per_cam = {"left_wrist_camera": {"result": "success"},
               "right_wrist_camera": {"result": "failed"}}
    d.check_inserted = lambda b, a, cam=None, epoch=0: per_cam[cam]
    v = make_verifiers(table_z=0.065, artifacts_dir=INSERT_DIR,
                       class_name="VialsInsertVerifiers", side="right", detector=d,
                       stand_region=(0.287, 0.370, 0.191, 0.319),
                       release_z_band=(0.205, 0.270), min_hold_frames=5,
                       open_ramp_ticks=6, after_delay_frames=3, drop_frames=6,
                       confirm_cameras=["left_wrist_camera", "right_wrist_camera"])
    r = run_episode(v)
    assert r.get("failure") and not r["success"] and not r["unrecoverable"], \
        "failed(any cam) must win -> episode FAILURE"
    print("PASS two-wrist combine: failed beats success (-> FAILURE)")

    r = run_episode(mk_verifier({"result": "failed"}))
    assert r.get("failure") and not r["success"] and not r["unrecoverable"]
    print("PASS verifier FAILURE (not Call-Human) on pair-confirmed vial fall")

    # not_done past all retries = FAILURE (user spec 2026-07-31: released but the
    # stand count did not increase -> episode ENDS, reward 0)
    v = mk_verifier({"result": "not_done"})                # retries=1 in tests
    r = run_episode(v, verdicts_after_release=10)
    assert r is not None and r.get("failure") and not r["success"] and not r["unrecoverable"]
    # one-shot after failure: a fresh episode through the SAME verifier still arms
    r = run_episode(v, verdicts_after_release=10)
    assert r is not None and r.get("failure"), "verifier must re-arm after a failure"
    print("PASS verifier FAILURE on not_done past retries (release, no new vial)")

    import time
    v = mk_verifier(None)                                  # confirm outage
    r = run_episode(v, verdicts_after_release=5)
    assert not r["success"]
    time.sleep(0.25)                                       # past confirm_timeout_s
    r = v.evaluate(obs(width=0.95, xyz=(0.33, 0.25, 0.27)))
    assert r["success"] and "proprio-only" in r["reason"]
    print("PASS verifier lenient proprio success on confirm outage")

    r = run_episode(mk_verifier({"result": "cannot_tell"}))
    assert r["success"] and "cannot_tell" in r["reason"]
    print("PASS verifier lenient success on cannot_tell")

    # ---- verifier: release OUTSIDE the pose gates STILL reaches the ER judge
    # (2026-08-03, ep0000: a 3 mm region miss silently ate a real success) -----
    r = run_episode(mk_verifier({"result": "success"}), xy=(0.18, -0.05),
                    verdicts_after_release=10)
    assert r is not None and r["success"], \
        "off-gate release must still be adjudicated by the pair check"
    r = run_episode(mk_verifier({"result": "not_done"}), xy=(0.18, -0.05),
                    verdicts_after_release=10)
    assert r is not None and r.get("failure"), "off-gate release + no new vial = FAILURE"
    print("PASS every release reaches the ER judge (pose gates are soft metadata)")

    # ---- selector: SAM3 stand-visibility confirm (user 2026-07-31) ----------
    def mk_sel(stand):
        return make_selector(table_z=0.065, artifacts_dir=INSERT_DIR,
                             class_name="VialsInsertSelector", side="right",
                             start_frames=3, stand_region=(0.287, 0.370, 0.191, 0.319),
                             approach_z_min=0.25, approach_z_max=0.42,
                             detector=StubDetector(stand_visible=stand))
    s = mk_sel(True)
    s.select(obs(), "rl")
    assert any(s.select(obs(width=0.03, xyz=(0.33, 0.25, 0.32)), "vla") == "rl"
               for _ in range(4))
    s = mk_sel(False)
    s.select(obs(), "rl")
    assert all(s.select(obs(width=0.03, xyz=(0.33, 0.25, 0.32)), "vla") is None
               for _ in range(6)), "stand not visible must block entry"
    s = mk_sel(None)
    s.select(obs(), "rl")
    assert all(s.select(obs(width=0.03, xyz=(0.33, 0.25, 0.32)), "vla") is None
               for _ in range(6)), "no stand verdict yet must block (mandatory image)"
    print("PASS selector wrist stand-confirm: visible fires / blind or pending blocks")

    # ---- verifier: mid-carry slip -> FAILURE (episode ends, VLA continues) --
    v = mk_verifier({"result": "success"})
    for _ in range(8):
        v.evaluate(obs(width=0.03, xyz=(0.33, 0.25, 0.30)))
    r = None
    for _ in range(10):                                    # width collapses (air-close)
        r = v.evaluate(obs(width=0.005, effort=0.02, xyz=(0.33, 0.25, 0.30)))
        if r.get("failure"):
            break
    assert r.get("failure") and "slipped" in r["reason"] and not r["unrecoverable"]
    print("PASS sustained hold loss without release -> vial slipped (FAILURE)")

    # ---- verifier: start_ok = open gripper (scene-ready gate) --------------
    v = mk_verifier(None)
    assert v.evaluate(obs(width=0.95))["start_ok"]
    assert not v.evaluate(obs(width=0.03))["start_ok"]
    v._reset()
    print("PASS start_ok open-gripper gate + _reset available")

    # ---- left-arm construction (RL3 readiness) ------------------------------
    v = make_verifiers(table_z=0.065, artifacts_dir=INSERT_DIR,
                       class_name="VialsInsertVerifiers", side="left",
                       detector=StubDetector(None))
    assert v.side == "left" and v.region == v.STAND_REGION["left"]
    s = make_selector(table_z=0.065, artifacts_dir=INSERT_DIR,
                      class_name="VialsInsertSelector", side="left")
    assert s.side == "left" and s.region[2] < 0 < s.region[1]
    print("PASS left-side artifacts construct with left calibration")

    full_task_loop_tests()
    print("\nALL PASS")


def full_task_loop_tests():
    """FULL-TASK insert training loop (user 2026-07-31): aux frozen grasp + primary
    learning insert in one inference-profile loop; batch HUMAN restage per scene."""
    from limb.agents.policy_learning.subtask.mode_machine import SubtaskMode, SubtaskModeMachine
    from limb.agents.policy_learning.subtask.rl_policy_client import RLPolicyClient
    from limb.agents.policy_learning.subtask.subtask_rl_agent import SubtaskRLAgent

    class StubVLA:
        def act(self, obs):
            return {"left": {"pos": np.zeros(7, np.float32)},
                    "right": {"pos": np.full(7, 0.5, np.float32)}}

        def action_spec(self):
            return None

        def reset(self):
            pass

    class FlagVerifiers:
        def __init__(self):
            self.start_ok = False
            self.success = False
            self.failure = False
            self.unrecoverable = False

        def evaluate(self, obs):
            return {"start_ok": self.start_ok, "success": self.success,
                    "failure": self.failure, "unrecoverable": self.unrecoverable}

        def _reset(self):
            pass

    class FlagSelector:
        def __init__(self):
            self.fire = False

        def select(self, obs, mode):
            return "rl" if (self.fire and mode == "vla") else None

    class StubCollector:
        def __init__(self):
            self.started = self.steps = self.transitions = self.ended = 0

        def start_episode(self):
            self.started += 1

        def record_step(self, *a, **k):
            self.steps += 1

        def record_rl_transition(self, *a, **k):
            self.transitions += 1

        def end_episode(self):
            self.ended += 1

        def note_cause(self, c):
            pass

        def stamp_terminal(self, **k):
            pass

    class StubEscalator:
        def __init__(self):
            self.calls = []

        def call_human(self, obs, reason=""):
            self.calls.append(reason)

    m = SubtaskModeMachine(profile="inference", min_mode_steps=2, cooldown_steps=0)
    pv, av = FlagVerifiers(), FlagVerifiers()
    ps, as_ = FlagSelector(), FlagSelector()
    col, esc = StubCollector(), StubEscalator()
    ag = SubtaskRLAgent(vla=StubVLA(), rl_client=RLPolicyClient(mode="dummy", action_dim=7),
                        machine=m, verifiers=pv, reset_policy=None, selector=ps,
                        aux_selector=as_, aux_verifiers=av,
                        aux_rl_client=RLPolicyClient(mode="dummy", action_dim=7),
                        collector=col, escalator=esc,
                        full_task_training=True, vials_per_scene=2,
                        human_reset_only=False, human_resume_frames=3,
                        skill_label="rl2_insert", aux_skill_label="rl1_grasp",
                        controlled_indices=list(range(7, 14)), left_mode="vla")

    def tick(n=1):
        for _ in range(n):
            ag.act({})

    assert m.mode == SubtaskMode.VLA
    # 1) aux (grasp) entry has priority; aux segment is NOT collected
    as_.fire = True
    tick(4)
    assert m.mode == SubtaskMode.RL and ag._active_aux
    assert col.started == 0 and col.steps == 0
    assert ag.phase_name() == "rl1_grasp" and ag.act({})["_mode"] == "rl1_grasp"
    print("PASS full-task: aux entry -> RL (labelled rl1_grasp), not collected")

    as_.fire = False
    av.success = True                        # frozen grasp succeeded, vial held
    tick(3)
    av.success = False
    assert m.mode == SubtaskMode.VLA and col.started == 0 and col.transitions == 0
    print("PASS full-task: aux success -> VLA, nothing pushed")

    # 2) primary (insert) entry: collected episode; success counts toward the scene
    ps.fire = True
    tick(4)
    ps.fire = False
    assert m.mode == SubtaskMode.RL and not ag._active_aux and col.started == 1
    assert col.steps > 0
    assert ag.phase_name() == "rl2_insert" and ag.act({})["_phase"] == "rl2_insert"
    pv.success = True
    tick(3)
    pv.success = False
    assert m.mode == SubtaskMode.VLA and ag._scene_successes == 1
    print("PASS full-task: primary success collected, scene count 1")

    # 3) second primary success exhausts the 2-vial scene -> HUMAN batch restage
    ps.fire = True
    tick(4)
    ps.fire = False
    pv.success = True
    tick(3)
    pv.success = False
    assert m.mode == SubtaskMode.HUMAN and ag._scene_successes == 2
    assert any("restage ALL" in c for c in esc.calls)
    print("PASS full-task: scene exhausted -> HUMAN batch restage (Call-Human once)")

    # 4) HUMAN resume -> VLA with a fresh scene counter
    pv.start_ok = True
    tick(8)
    assert m.mode == SubtaskMode.VLA and ag._scene_successes == 0
    print("PASS full-task: HUMAN resume -> VLA, scene counter reset")

    # 5) aux timeout -> VLA, still nothing collected
    as_.fire = True
    tick(4)
    as_.fire = False
    assert ag._active_aux
    started_before = col.started
    ag.aux_rl_timeout_steps = 3
    tick(6)
    assert m.mode == SubtaskMode.VLA and col.started == started_before
    print("PASS full-task: aux timeout -> VLA, not collected")

    # 6) primary FAILURE (released, no new vial) -> reward-0 episode end -> VLA
    ps.fire = True
    tick(4)
    ps.fire = False
    assert m.mode == SubtaskMode.RL and not ag._active_aux
    ended_before = col.ended + (1 if ag._end_episode_pending else 0)
    pv.failure = True
    tick(3)
    pv.failure = False
    assert m.mode == SubtaskMode.VLA and ag._scene_successes == 0
    assert col.ended + (1 if ag._end_episode_pending else 0) > ended_before
    print("PASS full-task: primary FAILURE -> reward-0 episode, back to VLA")

    # 7) unrecoverable during primary RL escalates to HUMAN (full_task_training)
    ps.fire = True
    tick(4)
    ps.fire = False
    assert m.mode == SubtaskMode.RL and not ag._active_aux
    pv.unrecoverable = True
    tick(2)
    pv.unrecoverable = False
    assert m.mode == SubtaskMode.HUMAN and len(esc.calls) >= 2
    print("PASS full-task: primary unrecoverable -> HUMAN escalation (not eval bypass)")


if __name__ == "__main__":
    main()
