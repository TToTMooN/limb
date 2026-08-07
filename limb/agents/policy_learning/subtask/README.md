# subtask — Sub-Task RL plumbing for limb (VLA-RL-AutoReset loop)

Real-robot integration for sub-task online RL on YAM, per
`SubRL-VLA/limb_subtask_rl_integration.md`. limb stays inference-only; this is an
additive subpackage.

## Modules

| File | Role |
|---|---|
| `mode_machine.py` | `SubtaskModeMachine` — VLA/RL/RESET/PAUSED/HUMAN/TERMINAL; training + inference transition profiles; oscillation guardrails; human-pedal override (modeled on `dagger/phase.py`). |
| `rl_policy_client.py` | `RLPolicyClient` (limb `PolicyClient`) — `dummy` (zero residual) for plumbing; `http` to the openpi-RLT actor (:9101). |
| `subtask_rl_agent.py` | `SubtaskRLAgent` (limb `Agent`) — frozen VLA + RL **residual** over `controlled_indices` (right-arm [7:14] single-arm, or [0:14] dual-arm); mode machine + verifiers + reset + selector; stamps + records each step. |
| `interfaces.py` | `Selector`/`ResetPolicy`/`SubtaskVerifiers` protocols + runnable defaults (`HoldReset`, `NullSelector`, `ScriptedVerifiers`, `CallableVerifiers`). The coding-agent harness authors the real perception versions. |
| `rollout_collector.py` | `RolloutCollector` — logs the **3 camera streams + states** (the coding-agent data) + RL fields (mode/reward/residual/source + right-arm-dominance audit); assembles `(s,a,r,s2,done)` for the RLT replay. |
| `test_subtask_offline.py` | Off-robot validation (13 assertions): transitions + single/dual-arm composition. |

## How the coding agents fit (CaP-X-style)

The `RolloutCollector` writes the online data — head/left_wrist/right_wrist frames +
robot states + outcomes — which the coding agents (Codex / Claude Code, in
`SubRL-VLA/coding_agent/`) read to author the **verifiers / reset / selector** as
code-as-policy (the perception predicates that read the same obs at runtime). This
is the CaP-X pattern: the agent reads the obs surface and emits the policy code.

## Validated off-robot · TODO for on-robot

Validated: mode transitions (RESET→RL→RESET-on-success, RL→HUMAN-on-unrecoverable),
residual composition (single/dual-arm), collector journal (3-cam + state + RL fields).

On-robot dry-run config: `configs/yam_subtask_rl_grasp.yaml` (dummy RL + scripted
verifier → the RESET↔RL loop cycles autonomously). Before running: fill camera
serials + the pi0.5 checkpoint server + the YamPolicyAgent transforms; then replace
the scripted/Hold/Null placeholders with the coding-agent-authored perception. The
HUMAN path needs the leader arms + a phase trigger (as in `yam_dagger_*`).
