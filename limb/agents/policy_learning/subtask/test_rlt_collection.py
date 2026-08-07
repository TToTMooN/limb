"""Offline proof that we TRULY collect RL rollouts for online RL training (review C2/C3/H9).

Simulates the autonomous training cycle with stubs (no robot, no servers):
  RESET -> RL (chunk-boundary actor queries) -> verifier SUCCESS (reward 1)
        -> episode closed with a done=True transition whose rewards contain the 1
        -> back to RESET (inverse policy; NO human) -> next RL rollout
plus the degrading path: unrecoverable -> Call Human fires ONCE -> episode closed reward 0.

Run:  uv run python limb/agents/policy_learning/subtask/test_rlt_collection.py
"""
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> policy_learning/

from subtask.mode_machine import SubtaskMode, SubtaskModeMachine
from subtask.rl_policy_client import RLPolicyClient
from subtask.rollout_collector import RolloutCollector
from subtask.subtask_rl_agent import SubtaskRLAgent

PASS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    PASS.append(bool(cond))


class StubVLA:
    """Frozen-VLA stand-in: constant actions + RLT features (z_rl, 50-step ref chunk, cursor)."""
    def __init__(self):
        self.resets = 0
        self._cursor = 0

    def act(self, obs):
        a = {"left": {"pos": np.zeros(7, np.float32)},
             "right": {"pos": np.full(7, 0.5, np.float32)},
             "_z_rl": np.ones(16, np.float32),
             "_ref_chunk": np.tile(np.arange(14, dtype=np.float32), (50, 1)),
             "_ref_cursor": self._cursor}
        self._cursor = (self._cursor + 1) % 50
        return a

    def reset(self):
        self.resets += 1
        self._cursor = 0

    def action_spec(self):
        return {}


class ScriptVerifier:
    """success at a scripted tick; unrecoverable at another (after which start_ok stays
    False — a wrecked scene isn't start-ready until a human actually restores it)."""
    def __init__(self, success_at=None, unrecoverable_at=None):
        self.n = 0
        self.success_at = success_at
        self.unrecoverable_at = unrecoverable_at
        self._wrecked = False

    def evaluate(self, obs):
        self.n += 1
        if self.unrecoverable_at is not None and self.n == self.unrecoverable_at:
            self._wrecked = True
        return {"start_ok": not self._wrecked,
                "success": self.success_at is not None and self.n == self.success_at,
                "unrecoverable": (self.unrecoverable_at is not None
                                  and self.n == self.unrecoverable_at),
                "reason": "vial fell on the floor" if self.unrecoverable_at else ""}


class QuickReset:
    def __init__(self):
        self._t = 0

    def act(self, obs):
        self._t += 1
        return {}

    def done(self, obs):
        return self._t >= 2

    def reset(self):
        self._t = 0


class CountEscalator:
    def __init__(self):
        self.calls = []

    def call_human(self, obs, reason):
        self.calls.append(reason)


OBS = {"left": {"joint_pos": np.zeros(6), "gripper_pos": [0.1]},
       "right": {"joint_pos": np.ones(6), "gripper_pos": [2.0]}}


def make_agent(verifier, collector, escalator, C=5):
    return SubtaskRLAgent(
        vla=StubVLA(),
        rl_client=RLPolicyClient(mode="dummy", action_dim=7),   # has infer_rlt -> chunk path
        machine=SubtaskModeMachine(profile="training", min_mode_steps=1, cooldown_steps=0),
        verifiers=verifier,
        reset_policy=QuickReset(),
        escalator=escalator,
        collector=collector,
        chunk_len=C,
    )


def main():
    tmp = tempfile.mkdtemp(prefix="rlt_collect_")

    # ---- 1) SUCCESS path: reward-1 transition must land in the collector -------
    col = RolloutCollector(out_dir=tmp + "/succ")           # no push_url: keep local
    esc = CountEscalator()
    ag = make_agent(ScriptVerifier(success_at=16), col, esc, C=5)
    modes = []
    for _ in range(30):
        modes.append(ag.act(OBS)["_mode"])
    eps = sorted(pathlib.Path(tmp + "/succ").glob("*/ep*/meta.json"))
    check("success episode written", len(eps) >= 1)
    import json
    meta = json.loads(eps[0].read_text())
    check("episode outcome=success", meta["outcome"] == "success")
    check("episode has RL transitions", meta["rl_transitions"] >= 1)
    check("episode has a REWARDED transition", meta["rewarded_transitions"] >= 1)
    journal = [json.loads(l) for l in (eps[0].parent / "rl_journal.jsonl").read_text().splitlines()]
    rl_steps = [r for r in journal if r["mode"] == "rl"]
    check("terminal reward stamped on last RL step", rl_steps and rl_steps[-1]["reward"] == 1.0
          and rl_steps[-1].get("terminal"))
    check("NO human call on the success path", len(esc.calls) == 0)
    # autonomous cycle: ... rl (success) -> reset (inverse policy, vial held) -> rl again
    first_rl_end = max(i for i, m in enumerate(modes) if m == "rl" and "reset" in modes[i + 1:i + 3])
    check("auto-reset followed success (no human)", "reset" in modes[first_rl_end:first_rl_end + 3])
    check("second RL rollout started autonomously", ag.machine.mode == SubtaskMode.RL
          and modes.count("rl") > modes.index("reset"))

    # ---- 2) chunk math: transitions have rewards of length C -----------------------
    tr_file = None
    # re-run capturing transitions in-memory via a fresh collector w/o disk push
    col2 = RolloutCollector(out_dir=tmp + "/succ2")
    captured = []
    col2.record_rl_transition, _orig = (lambda f, r, d, nf: captured.append((f, r, d, nf))), col2.record_rl_transition
    ag2 = make_agent(ScriptVerifier(success_at=14), col2, CountEscalator(), C=5)
    for _ in range(25):
        ag2.act(OBS)
    check("chunk transitions captured", len(captured) >= 1)
    check("rewards vector has length C",
          all(len(np.asarray(r)) == 5 for _f, r, _d, _nf in captured))
    terminal = [(f, r, d, nf) for f, r, d, nf in captured if d]
    check("terminal transition done=True with reward 1 inside",
          any(float(np.max(np.asarray(r))) == 1.0 for _f, r, _d, _nf in terminal))
    feats = captured[0][0]
    check("transition carries z_rl + ref_chunk + action_chunk",
          feats.get("z_rl") is not None and feats.get("ref_chunk") is not None
          and feats.get("action_chunk") is not None)
    check("ref_chunk is right-arm 7-D x C", np.asarray(feats["ref_chunk"]).shape == (5, 7))

    # ---- 3) UNRECOVERABLE path: Call Human once, episode closed reward 0 ----------
    col3 = RolloutCollector(out_dir=tmp + "/unrec")
    esc3 = CountEscalator()
    ag3 = make_agent(ScriptVerifier(unrecoverable_at=18), col3, esc3, C=5)
    for _ in range(30):
        ag3.act(OBS)
    check("Call Human fired exactly once", len(esc3.calls) == 1)
    check("Call Human carries the reason", "floor" in esc3.calls[0])
    check("agent parked in HUMAN", ag3.machine.mode == SubtaskMode.HUMAN)
    eps3 = sorted(pathlib.Path(tmp + "/unrec").glob("*/ep*/meta.json"))
    meta3 = json.loads(eps3[0].read_text()) if eps3 else {}
    check("unrecoverable episode written with outcome", meta3.get("outcome") == "unrecoverable")

    # ---- 4) PAUSED resume (review H8) ---------------------------------------------
    m = SubtaskModeMachine(profile="training", min_mode_steps=1, cooldown_steps=0)
    m.request_human("pause_resume"); m.consume_human()
    check("pedal pauses", m.mode == SubtaskMode.PAUSED)
    m.request_human("pause_resume"); m.consume_human()
    check("pedal RESUMES to previous mode", m.mode == SubtaskMode.RESET)

    # ---- 4b) VLA APPROACH flow: RESET -> VLA -> selector 'rl' -> RL ---------------
    class AboveVialSelector:
        def __init__(self, fire_after=8):
            self.n = 0; self.fire_after = fire_after
        def select(self, obs, mode):
            if mode == "vla":
                self.n += 1
                if self.n >= self.fire_after:
                    return "rl"
            return None

    ag5 = make_agent(ScriptVerifier(success_at=10**9), RolloutCollector(out_dir=tmp + "/appr"),
                     CountEscalator(), C=5)
    ag5.use_vla_approach = True
    ag5.selector = AboveVialSelector()
    ag5.rl_timeout_steps = 5                     # force episode ends so the loop cycles
    seq = [ag5.act(OBS)["_mode"] for _ in range(60)]
    check("approach: RESET stages first", seq[0] == "reset")
    check("approach: VLA phase runs after RESET", "vla" in seq)
    check("approach: selector hands off to RL", "rl" in seq and seq.index("vla") < seq.index("rl"))
    # user spec 2026-07-06 22:52: VLA drives ONLY the FIRST approach — after the first
    # RL entry the loop must cycle RL <-> RESET with NO further VLA phase.
    first_rl = seq.index("rl")
    check("VLA NEVER returns after the first RL entry", "vla" not in seq[first_rl:])
    check("loop cycles RL <-> RESET after the first approach",
          seq[first_rl:].count("reset") >= 2 and seq[first_rl:].count("rl") >= 6)

    # ---- 4c2) RL entry BLOCKED until the verifier confirms the START STATE (user
    # rule 2026-07-06 23:0x): reset done() alone is not enough — a FALLEN vial is
    # "visible" but is not a valid bottleneck start; the loop must hold in RESET.
    class FallenVerifier(ScriptVerifier):
        def evaluate(self, obs):
            v = super().evaluate(obs)
            v["vial_visible"] = True
            v["vial_fallen"] = True            # lying on its side in the head camera
            return v

    ag7 = make_agent(FallenVerifier(), RolloutCollector(out_dir=tmp + "/fallen"),
                     CountEscalator(), C=5)
    modes7 = [ag7.act(OBS)["_mode"] for _ in range(30)]
    check("fallen start state: RL (and VLA) never entered — parked in RESET",
          "rl" not in modes7 and "vla" not in modes7 and modes7[-1] == "reset")

    # ---- 4c) reset FAILS -> ONE attempt only, then Call-Human (user rule 2026-07-07:
    # the human demos place the vial ONCE; the reset must never re-try a placement) --
    class FailReset:
        def __init__(self):
            self.failed = True
            self.resets = 0

        def act(self, obs):
            return {}

        def done(self, obs):
            return False

        def reset(self):
            self.resets += 1
            self.failed = True                      # keeps failing every attempt

    OBS_HOLD = {"left": {"joint_pos": np.zeros(6), "gripper_pos": [0.1]},
                "right": {"joint_pos": np.ones(6), "gripper_pos": [0.05]}}  # held band
    esc6 = CountEscalator()
    ag6 = make_agent(ScriptVerifier(success_at=10**9),
                     RolloutCollector(out_dir=tmp + "/failhold"), esc6, C=5)
    ag6.reset_policy = FailReset()
    for _ in range(20):
        ag6.act(OBS_HOLD)
    check("failed reset: SINGLE attempt (no re-tries)", ag6.reset_policy.resets <= 1)
    check("failed reset: Call-Human fired once", len(esc6.calls) == 1)
    check("failed reset: parked in HUMAN (vial still held, gripper closed)",
          ag6.machine.mode == SubtaskMode.HUMAN)

    # ---- 4f) HUMAN-RESET-ONLY mode (user 2026-07-08): no RESET phase at all —
    # boot VLA -> RL, success opens the gripper and goes to HUMAN, resume -> RL.
    class GoodStartVerifier(ScriptVerifier):
        """start_ok/visible/standing always true so the HUMAN resume gate can pass."""
        def evaluate(self, obs):
            v = super().evaluate(obs)
            v["vial_visible"] = True
            v["vial_standing"] = True
            return v

    ag10 = make_agent(GoodStartVerifier(success_at=12), RolloutCollector(out_dir=tmp + "/honly"),
                      CountEscalator(), C=5)
    ag10.human_reset_only = True
    ag10.use_vla_approach = True
    ag10.machine.reset(SubtaskMode.VLA)            # what __post_init__ does when configured
    ag10.selector = AboveVialSelector(fire_after=4)
    ag10.human_resume_frames = 5                   # quick resume for the test
    seq, human_grips = [], []
    for _ in range(60):
        a = ag10.act(OBS)
        seq.append(a["_mode"])
        if a["_mode"] == "human" and "right" in a:
            human_grips.append(float(a["right"]["pos"][6]))
    check("human-only: NO reset phase ever", "reset" not in seq)
    check("human-only: boot VLA -> RL -> success -> HUMAN",
          seq[0] == "vla" and "rl" in seq and "human" in seq
          and seq.index("rl") < seq.index("human"))
    check("human-only: gripper OPENS immediately on the success handoff",
          bool(human_grips) and human_grips[0] > 1.1)
    check("human-only: resumed straight back to RL",
          "rl" in seq[seq.index("human"):])
    ep10 = sorted(pathlib.Path(tmp + "/honly").glob("*/ep*/meta.json"))
    m10 = json.loads(ep10[0].read_text()) if ep10 else {}
    check("human-only: success episode recorded with a rewarded transition",
          m10.get("outcome") == "success" and m10.get("rewarded_transitions", 0) >= 1)

    # ---- 4e) vial reported FALLEN during the RL rollout -> rollout CONTINUES
    # (rule deleted, user 2026-07-08: SAM3 false-fallens were terminating episodes
    # before the policy reached the vial). The episode must run to its timeout
    # (reward 0 -> HUMAN), and the fallen flag must still block the HUMAN resume.
    class FallenDuringRL(ScriptVerifier):
        def evaluate(self, obs):
            v = super().evaluate(obs)
            v["vial_visible"] = True
            v["vial_fallen"] = True
            v["start_ok"] = True                   # resume gate would otherwise pass
            return v

    esc9 = CountEscalator()
    ag9 = make_agent(FallenDuringRL(success_at=10**9),
                     RolloutCollector(out_dir=tmp + "/rlfall"), esc9, C=5)
    ag9.human_reset_only = True                    # production loop config (timeout -> HUMAN)
    ag9.rl_timeout_steps = 30                      # small timeout for the test
    ag9.machine.reset(SubtaskMode.RL)              # already in an RL rollout
    for _ in range(20):
        ag9.act(OBS)
    check("fallen during RL: rollout NOT terminated by the detector (rule deleted)",
          ag9.machine.mode == SubtaskMode.RL)
    for _ in range(25):
        ag9.act(OBS)
    check("fallen during RL: episode ends via the RL TIMEOUT -> HUMAN",
          ag9.machine.mode == SubtaskMode.HUMAN)
    import glob as _glob
    metas9 = sorted(_glob.glob(tmp + "/rlfall/*/ep*/meta.json"))
    m9 = json.loads(open(metas9[0]).read()) if metas9 else {}
    check("fallen during RL: timeout episode closed with ZERO rewarded transitions",
          bool(metas9) and m9.get("rewarded_transitions") == 0)
    for _ in range(30):                            # fallen persists -> resume must stay blocked
        ag9.act(OBS)
    check("fallen vial still blocks the HUMAN->RL resume gate",
          ag9.machine.mode == SubtaskMode.HUMAN)

    # ---- 4f) INFERENCE profile: RL timeout hands back to VLA (full-task eval,
    # user 2026-07-13) — the old RESET proposal had no edge in the inference
    # profile, was silently rejected, and the agent re-finalized the episode
    # every tick.
    ag11 = SubtaskRLAgent(
        vla=StubVLA(), rl_client=RLPolicyClient(mode="dummy", action_dim=7),
        machine=SubtaskModeMachine(profile="inference", min_mode_steps=1, cooldown_steps=0),
        verifiers=ScriptVerifier(success_at=10**9),
        reset_policy=QuickReset(), escalator=CountEscalator(),
        collector=RolloutCollector(out_dir=tmp + "/infer_to"), chunk_len=5)
    ag11.rl_timeout_steps = 20
    ag11.machine.reset(SubtaskMode.RL)
    for _ in range(40):
        ag11.act(OBS)
    check("inference timeout: agent returned to VLA (not wedged in RL/RESET)",
          ag11.machine.mode == SubtaskMode.VLA)
    metas11 = sorted(pathlib.Path(tmp + "/infer_to").glob("*/ep*/meta.json"))
    check("inference timeout: exactly ONE episode finalized (no per-tick churn)",
          len(metas11) == 1)
    import json as _json11
    check("inference timeout: episode outcome=timeout reward 0",
          bool(metas11) and _json11.loads(metas11[0].read_text())["outcome"] == "timeout")

    # ---- 4h) NO HUMAN PHASE at inference (user 2026-07-14): unrecoverable during
    # an eval RL attempt ends the episode (reward 0) and returns to VLA — never
    # HUMAN, never Call-Human. The operator labels/restages via the keyboard.
    esc13 = CountEscalator()
    ag13 = SubtaskRLAgent(
        vla=StubVLA(), rl_client=RLPolicyClient(mode="dummy", action_dim=7),
        machine=SubtaskModeMachine(profile="inference", min_mode_steps=1, cooldown_steps=0),
        verifiers=ScriptVerifier(unrecoverable_at=8),
        reset_policy=QuickReset(), escalator=esc13,
        collector=RolloutCollector(out_dir=tmp + "/infer_unrec"), chunk_len=5)
    ag13.machine.reset(SubtaskMode.RL)
    for _ in range(20):
        ag13.act(OBS)
    check("eval unrecoverable: NO HUMAN phase (agent back in VLA)",
          ag13.machine.mode == SubtaskMode.VLA)
    check("eval unrecoverable: Call-Human NEVER fired", len(esc13.calls) == 0)
    metas13 = sorted(pathlib.Path(tmp + "/infer_unrec").glob("*/ep*/meta.json"))
    check("eval unrecoverable: episode closed with outcome=unrecoverable reward 0",
          bool(metas13) and _json11.loads(metas13[0].read_text())["outcome"] == "unrecoverable")

    # ---- 4g) foot pedal pause/resume (user 2026-07-14): a phase_trigger's listener
    # feeds machine.request_human("pause_resume"); act() consumes it — PAUSED holds,
    # a second press resumes the previous mode. Works in BOTH profiles.
    class StubPedal:
        def start(self, events):
            self.events = events               # capture the agent's inbox adapter

    for prof in ("training", "inference"):
        ped = StubPedal()
        ag12 = SubtaskRLAgent(
            vla=StubVLA(), rl_client=RLPolicyClient(mode="dummy", action_dim=7),
            machine=SubtaskModeMachine(profile=prof, min_mode_steps=1, cooldown_steps=0),
            verifiers=ScriptVerifier(success_at=10**9),
            reset_policy=QuickReset(), escalator=CountEscalator(),
            collector=RolloutCollector(out_dir=tmp + f"/pedal_{prof}"),
            chunk_len=5, phase_trigger=ped)
        ag12.machine.reset(SubtaskMode.RL)
        ag12.act(OBS)
        ped.events.request_transition("pause_resume")      # pedal press (listener thread)
        ag12.act(OBS)
        check(f"pedal pauses the {prof} loop", ag12.machine.mode == SubtaskMode.PAUSED)
        ped.events.request_transition("pause_resume")      # second press
        ag12.act(OBS)
        check(f"pedal resumes the {prof} loop out of PAUSED",
              ag12.machine.mode != SubtaskMode.PAUSED)
        check(f"pedal 'correction' is ignored in the {prof} loop",
              ped.events.request_transition("correction") is False)

    # ---- 4i) EVAL resume-after-pause is ALWAYS VLA (user 2026-07-14): pausing at
    # inference means the operator restaged — the arms are back at the initial
    # state, so a paused mid-episode RL context must not resume. Training keeps
    # resume-to-previous (covered by test 4).
    m_ev = SubtaskModeMachine(profile="inference", min_mode_steps=1, cooldown_steps=0)
    m_ev.reset(SubtaskMode.RL)                     # paused mid-RL-episode
    m_ev.request_human("pause_resume"); m_ev.consume_human()
    check("eval pause from RL parks in PAUSED", m_ev.mode == SubtaskMode.PAUSED)
    m_ev.request_human("pause_resume"); m_ev.consume_human()
    check("eval resume goes to VLA (never back into the paused RL episode)",
          m_ev.mode == SubtaskMode.VLA)

    # ---- 4d) HUMAN phase hard cap (user rule 2026-07-07: ~15 s, never stick) --------
    esc8 = CountEscalator()
    ag8 = make_agent(ScriptVerifier(unrecoverable_at=5), RolloutCollector(out_dir=tmp + "/hcap"),
                     esc8, C=5)
    ag8.human_timeout_steps = 8                    # tiny cap for the test
    for _ in range(25):
        ag8.act(OBS)
    check("HUMAN timeout: loop left HUMAN without operator action",
          ag8.machine.mode != SubtaskMode.HUMAN)

    # ---- 5) RESET->HUMAN edge exists (review H7) -----------------------------------
    m2 = SubtaskModeMachine(profile="training", min_mode_steps=1, cooldown_steps=0)
    check("RESET->HUMAN escalation allowed", m2.propose(SubtaskMode.HUMAN, force=True) is not None)

    print("\n" + ("ALL PASS" if all(PASS) else f"{PASS.count(False)} FAILURES"))
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
