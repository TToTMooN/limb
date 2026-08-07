"""Off-robot validation of the sub-task plumbing: mode transitions + residual
composition, with a stub VLA + dummy RL client. No robot/limb-internals needed.

Run:  <py-with-numpy> limb/limb/agents/policy_learning/subtask/test_subtask_offline.py
"""
import pathlib
import sys

import numpy as np

# import the subtask package standalone (avoid the limb/__init__ chain)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from subtask.interfaces import HoldReset
from subtask.mode_machine import SubtaskMode, SubtaskModeMachine
from subtask.rl_policy_client import RLPolicyClient
from subtask.subtask_rl_agent import DUAL_INDICES, RIGHT_INDICES, SubtaskRLAgent


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
        self.start_ok = True; self.success = False; self.unrecoverable = False
    def evaluate(self, obs):
        return {"start_ok": self.start_ok, "success": self.success, "unrecoverable": self.unrecoverable}


def build(profile="training", ctrl=RIGHT_INDICES, left_mode="vla"):
    m = SubtaskModeMachine(profile=profile, min_mode_steps=2, cooldown_steps=0)
    v = FlagVerifiers()
    ag = SubtaskRLAgent(vla=StubVLA(), rl_client=RLPolicyClient(mode="dummy", action_dim=len(ctrl)),
                        machine=m, verifiers=v, reset_policy=HoldReset(settle=2),
                        controlled_indices=list(ctrl), left_mode=left_mode)
    return ag, v, m


def main():
    ok = True
    def check(name, cond):
        nonlocal ok; ok = ok and cond
        print(("PASS " if cond else "FAIL ") + name)

    # --- training: RESET -> RL ---
    ag, v, m = build()
    check("starts in RESET", m.mode == SubtaskMode.RESET)
    a = ag.act({}); check("reset action tagged", a["_mode"] == "reset")
    a = ag.act({})                      # reset done + start_ok -> RL
    a = ag.act({})
    check("RESET->RL after reset done", m.mode == SubtaskMode.RL)

    # --- RL composition: dummy residual=0 -> right == VLA.right, left == VLA.left ---
    check("RL: right == VLA.right (residual 0)", np.allclose(a["right"]["pos"], 0.5))
    check("RL: left == VLA.left", np.allclose(a["left"]["pos"], 0.0))
    check("RL: right_action_source", a["_right_action_source"] == "rl")

    # --- RL -> RESET on success (training) ---
    v.success = True; ag.act({}); v.success = False
    check("RL->RESET on success (training)", m.mode == SubtaskMode.RESET)

    # --- RL -> HUMAN on unrecoverable ---
    ag, v, m = build()
    for _ in range(4): ag.act({})       # get into RL
    check("in RL before unrecoverable", m.mode == SubtaskMode.RL)
    v.unrecoverable = True; ag.act({})
    check("RL->HUMAN on unrecoverable", m.mode == SubtaskMode.HUMAN)

    # --- dual-arm composition: residual length 14, both arms RL ---
    ag, v, m = build(ctrl=DUAL_INDICES)
    for _ in range(4): ag.act({})
    a = ag.act({})
    check("dual-arm: left_action_source rl", a["_left_action_source"] == "rl")
    check("dual-arm: controlled 14", len(a["_controlled_indices"]) == 14)

    # --- nonzero residual moves only controlled (right) dims ---
    ag, v, m = build()
    class FixedRL:
        def infer(self, obs): return {"actions": np.ones((1, 7), np.float32)}  # +1 residual
        def get_metadata(self): return {}
    ag.rl_client = FixedRL()
    for _ in range(4): ag.act({})
    a = ag.act({})
    check("residual moves right (0.5 + 0.1*1 pose)", np.isclose(a["right"]["pos"][0], 0.6, atol=1e-5))
    check("residual leaves left untouched", np.allclose(a["left"]["pos"], 0.0))

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
