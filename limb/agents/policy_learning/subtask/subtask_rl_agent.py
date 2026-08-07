"""SubtaskRLAgent — composite limb Agent for sub-task, sub-action-space online RL.

Composes a FROZEN VLA agent (pi0.5 — inference only, never trained: RLT-style) + an RL
actor client + the RESET<->RL<->HUMAN mode machine + the coding-agent artifacts
(verifiers = 0/1 reward, EAP inverse reset, selector for inference-time VLA<->RL).
Implements limb's ``Agent`` protocol (``act(obs) -> action``).

The autonomous training cycle (RoboClaw-style): most episodes need NO human —
  RESET (inverse policy places the held vial back at a random reachable xy)
  -> RL (pi0.5 + bounded right-arm residual grasps + stably lifts)
  -> verifier success -> RESET again (vial is HELD, exactly the inverse's precondition).
The human is called ONLY on a degrading/unrecoverable state (vial on the floor).

RL granularity is CHUNK-LEVEL to match openpi-RLT (review H9/H10): the actor is queried
once per ``chunk_len`` control steps with (z_rl, right-arm proprio, the ALIGNED right-arm
slice of the VLA reference chunk); the returned action chunk is executed step-by-step as
the residual; ONE transition per chunk (rewards (C,), sparse) goes to the collector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from limb.utils.portal_utils import remote

from .interfaces import (
    HoldReset,
    HumanEscalator,
    LoggingEscalator,
    NullSelector,
    ResetPolicy,
    Selector,
    SubtaskVerifiers,
)
from .mode_machine import SubtaskMode, SubtaskModeMachine

# 14-D residual bound: [joints x6, gripper] per arm (units: rad / gripper command).
# GRIPPER bound = 0 (user rule 2026-07-08): the EXECUTED gripper command is pure pi0.5 —
# run 164734 showed an actor gripper residual flickers the squeeze effort below LOAD_HOLD
# and the proprio success verifier misses real grasps. The actor still OUTPUTS a gripper
# residual and transitions store those REAL values (action_chunk is pre-bound unit
# residuals), so the gripper head keeps training; restore 0.5 here to re-enable execution.
DEFAULT_BOUND = np.array([0.1] * 6 + [0.0] + [0.1] * 6 + [0.0], dtype=np.float32)
RIGHT_INDICES = list(range(7, 14))
DUAL_INDICES = list(range(14))

# Obs keys the verifier/reset artifacts require (review C1: fail LOUDLY at startup
# instead of wedging silently when e.g. ee_pose was never populated by the launch path).
_REQUIRED_OBS_KEYS = [("right", "joint_pos"), ("right", "gripper_pos"),
                      ("right", "joint_eff"), ("right", "ee_pose")]


def _log(msg: str, warn: bool = False) -> None:
    try:
        from loguru import logger
        (logger.warning if warn else logger.info)("[SubtaskRLAgent] " + msg)
    except Exception:
        pass


@dataclass
class SubtaskRLAgent:
    vla: Any                                   # a limb Agent: act(obs)->{"left":{"pos"},"right":{"pos"}}
    rl_client: Any                             # RLPolicyClient (infer_rlt) or legacy (infer)
    machine: SubtaskModeMachine
    verifiers: SubtaskVerifiers
    reset_policy: ResetPolicy = field(default_factory=HoldReset)
    selector: Selector = field(default_factory=NullSelector)
    escalator: HumanEscalator = field(default_factory=LoggingEscalator)   # RoboClaw Call-Human
    controlled_indices: List[int] = field(default_factory=lambda: list(RIGHT_INDICES))
    left_mode: str = "vla"                     # "vla" | "hold"
    hold_pose_left: Optional[np.ndarray] = None
    bound: np.ndarray = field(default_factory=lambda: DEFAULT_BOUND.copy())
    collector: Optional[Any] = None            # RolloutCollector (journal + RLT transitions)
    chunk_len: int = 10                        # RLT chunk length C
    # Absolute-action clip = per-joint limits, NOT [-1,1] (review C5: limb actions are
    # raw joint radians + gripper in [0, 2.4]; a unit clip froze the gripper shut).
    # None = no absolute clip (residual is bounded anyway); set from the robot config.
    action_low: Optional[np.ndarray] = None    # (14,)
    action_high: Optional[np.ndarray] = None   # (14,)
    reset_timeout_steps: int = 900             # ~30 s @ 30 Hz: RESET escape hatch (review H14)
    # Episode structure (user spec 2026-07-06 22:52): the VLA drives ONLY THE FIRST
    # approach of the session (boot: RESET stages -> VLA approaches -> selector fires
    # above the vial -> RL). Once the RL loop has begun it NEVER returns to VLA during
    # training: the auto-reset ends retracted directly ABOVE the placed vial, so every
    # subsequent RL rollout is just grasp+lift (short-horizon). VLA<->RL switching is
    # an inference-time behavior.
    use_vla_approach: bool = False
    # HUMAN-RESET-ONLY loop (user mode 2026-07-08): NO auto-reset phase at all.
    # Boot: VLA first approach -> RL, then the loop is RL <-> HUMAN forever: EVERY
    # episode end (success reward 1 OR failure reward 0) hands the scene to the
    # human, who guarantees the sub-task start state; the verified resume (or the
    # 15 s cap) returns straight to RL. On SUCCESS the gripper OPENS immediately
    # (right-arm grasp sub-task: release the lifted vial for the human to re-stage).
    # False = the autonomous RL <-> RESET (<-> HUMAN) loop with the coding-agent reset.
    human_reset_only: bool = False
    # Where HUMAN resumes to (2026-07-31, insert sub-task's VLA-approach loop):
    # "auto" = the original rule (training -> RL directly; inference -> VLA).
    # "vla"  = training resumes to the VLA so pi0.5 re-does its reliable segment
    #          (grasp + carry) and the SELECTOR fires the sub-task entry EVERY
    #          episode — entry states then match eval's VLA-approach distribution.
    human_resume_to: str = "auto"

    # -- FULL-TASK INSERT TRAINING (user 2026-07-31): TWO RL skills in one loop --
    # The loop runs the whole task (inference profile: VLA <-> RL) with an AUX
    # frozen skill (the trained grasp RL1, deterministic actor) feeding the PRIMARY
    # learning skill (insert RL2, collected + pushed to replay). Scene = N vials on
    # the table; occupied stand holes accumulate across episodes — exactly the
    # empty-hole distribution pi0.5 fails on. HUMAN restages the WHOLE scene only
    # after vials_per_scene primary successes (batch reset) or on unrecoverable.
    aux_selector: Any = None        # aux entry gate (e.g. grasp strict SAM3 gate)
    aux_verifiers: Any = None       # aux exit: success/timeout -> VLA; NOT collected
    aux_rl_client: Any = None       # frozen actor service (--no-learner snapshot)
    aux_rl_timeout_steps: int = 300
    full_task_training: bool = False   # inference profile + HUMAN escalation + collection
    vials_per_scene: int = 0           # >0: batch-restage HUMAN after N primary successes
    # Phase labels for the RL mode (user 2026-08-03): the phase log / TUI / recorded
    # phase.npy say WHICH skill is running, e.g. "rl2_insert" / "rl1_grasp" instead
    # of a bare "rl". Defaults keep single-skill configs' phase.npy values unchanged.
    skill_label: str = "rl"            # PRIMARY (learning) skill's RL phase label
    aux_skill_label: str = "rl"        # AUX (frozen) skill's RL phase label
    approach_timeout_steps: int = 1800         # ~60 s of VLA approach without the selector firing -> Call-Human
    rl_timeout_steps: int = 400                # ~13 s of RL without success = FAILED episode (reward 0)
                                               # -> RESET re-stages and the loop retries. Short budget is right:
                                               # grasp+lift-a-little from a gripper already above the vial.
    open_gripper_on_human: bool = True         # H18: reopen the (empty) gripper on escalation
    # HUMAN resume gate: start_ok AND vial visible, SUSTAINED this many ticks (~2 s).
    # The bare start_ok gate resumed in <1 s after a failed reset (2026-07-06 21:25 run:
    # human -> reset -> vla with the vial on the floor) — a human never intervened.
    human_resume_frames: int = 60
    # HUMAN phase hard cap (user rule 2026-07-07: ~15 s, don't stick in HUMAN): after
    # this many ticks the loop forces RESET->RL regardless of the resume gate — the
    # RESET stage + RL-entry verification re-checks the scene anyway, and a still-bad
    # scene re-escalates (better a periodic re-check than an indefinitely parked loop).
    human_timeout_steps: int = 450
    gripper_open_cmd: float = 2.2
    # HUMAN-phase VIEW LIFT (user 2026-07-27): after RL -> HUMAN, servo the RIGHT
    # EE straight UP to this z (arm-base frame) so the wrist camera overlooks the
    # staging area — low RL terminal poses left it nearly on the table and vials
    # placed away from the gripper were invisible to the resume/selector checks.
    # xy + entry pose held; bounded DLS steps with command integration (stiction
    # rule 2026-07-06: never step from observed joints). 0 disables (frozen hold).
    human_lift_view_z: float = 0.28
    human_lift_step_rad: float = 0.02          # per-joint per-tick cap (~0.6 rad/s)
    human_lift_lead_rad: float = 0.08          # max commanded lead over observed joints
    fk: Optional[Any] = None                   # EEPoseInjector (review C1): fills right.ee_pose via FK
    # Always-on flight recorder: the agent subprocess's stderr is SWALLOWED by the TUI
    # (observed 2026-07-06: the cmd-obs delta logs never surfaced), so diagnostics are
    # appended to this file instead. tail -f it during a run.
    debug_file: str = "logs/subtask_debug.jsonl"
    debug_every: int = 15                      # ticks between records (~0.5 s @ 30 Hz)
    # Foot pedal -> PAUSED<->resume (user 2026-07-14: the TUI advertised '[L pedal]
    # pause/resume' but nothing ever produced pedal events in the SubRL loop — the
    # machinery lives in the DAgger agent). Wire a FootPedalPhaseTrigger here; its
    # listener thread feeds machine.request_human("pause_resume"), consumed at the
    # top of every _handle_transitions tick. None = no pedal (keyboard/TUI only).
    phase_trigger: Optional[Any] = None

    def __post_init__(self):
        self._ctrl = np.asarray(self.controlled_indices, dtype=int)
        self._v: Dict[str, Any] = {}
        self._last_residual = None
        self._prev_mode: Optional[SubtaskMode] = None
        self._validated = False
        self._warned_proprio = False
        self._end_episode_pending = False
        self._active_aux = False      # current RL segment runs the AUX (frozen) skill
        self._scene_successes = 0     # primary successes since the last scene restage
        self._human_ready = 0
        self._reset_retries = 0
        self._reset_attempt_start = 0
        self._rl_seen = False        # first RL entry ends the one-time VLA approach phase
        self._human_hold_q = None    # (lj6, rj6) frozen at HUMAN entry — the arm must
                                     # NOT move during HUMAN (user: frame-1890 pose drifted)
        self._human_open_gripper = False   # human_reset_only: success -> open immediately
        self._clear_chunk_state()
        if self.phase_trigger is not None:
            # Adapter: FootPedalPhaseTrigger.start() expects a DAggerEvents-like
            # inbox with request_transition(event) -> bool. Map the pause pedal to
            # the machine's human-override queue; 'correction' has no meaning here.
            machine = self.machine

            class _PedalInbox:
                def request_transition(self, event: str) -> bool:
                    if event == "pause_resume":
                        machine.request_human("pause_resume")
                        return True
                    return False

            try:
                self.phase_trigger.start(_PedalInbox())
                _log("foot pedal wired: pause/resume -> PAUSED<->previous mode")
            except Exception as e:  # pedal is OPTIONAL — never block the loop on it
                _log(f"foot pedal unavailable ({e}); pause pedal disabled", warn=True)
                self.phase_trigger = None
        if self.use_vla_approach:
            # Boot straight into the VLA approach (user rule 2026-07-07: the FIRST
            # phase is VLA, not a staging reset).
            self.machine.reset(SubtaskMode.VLA)

    def _clear_chunk_state(self) -> None:
        self._chunk_feats: Optional[Dict[str, Any]] = None    # feats of the chunk being executed
        self._cached_chunk: Optional[np.ndarray] = None       # (C, adim) actor output
        self._chunk_rewards: List[float] = []
        self._chunk_pos = 0

    # -- limb Agent protocol -------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        import time as _time
        _t = {}
        _t0 = _time.perf_counter()
        self.machine.tick()
        if self.fk is not None:                # fill right.ee_pose via FK BEFORE anything reads it
            obs = self.fk.inject(obs)
        # HD frame cache: launch ships a 448-px images["rgb_hd"] every ~15th tick
        # (policy rgb stays 224 every tick). Cache and re-attach it so PERCEPTION
        # (which prefers rgb_hd) always sees the latest HD view — staleness <=0.5 s,
        # far inside the ~2 s detection throttle.
        self._hd_cache = getattr(self, "_hd_cache", {})
        for _cam in ("right_wrist_camera", "head_camera", "left_wrist_camera"):
            try:
                imgs = obs[_cam]["images"]
                if "rgb_hd" in imgs:
                    self._hd_cache[_cam] = imgs["rgb_hd"]
                elif _cam in self._hd_cache:
                    imgs["rgb_hd"] = self._hd_cache[_cam]
            except Exception:
                pass
        # Depth mode (2026-08-05): stash the HD-tick depth frames into the shared
        # detectors (lingbot lane). The stash persists between HD ticks, mirroring
        # the rgb_hd cache above — so depth and rgb_hd stay from the SAME tick.
        try:
            from .perception import observe_depth as _observe_depth
            _observe_depth(obs)
        except Exception:
            pass
        _t["fk"] = _time.perf_counter() - _t0
        self._validate_obs_once(obs)
        _t1 = _time.perf_counter()
        self._handle_transitions(obs)          # verifier.evaluate + selector.select inside
        _t["transitions"] = _time.perf_counter() - _t1
        mode = self.machine.mode
        self._on_mode_entry(mode, obs)

        _t2 = _time.perf_counter()
        if mode == SubtaskMode.RESET:
            a = dict(self.reset_policy.act(obs))
        elif mode == SubtaskMode.RL:
            a_vla = self.vla.act(obs)
            _t["vla_act"] = _time.perf_counter() - _t2
            a = self._compose(a_vla, self._rl_step(obs, a_vla))
        elif mode == SubtaskMode.VLA:
            a = dict(self.vla.act(obs))
            _t["vla_act"] = _time.perf_counter() - _t2
            a.pop("_z_rl", None); a.pop("_ref_chunk", None); a.pop("_ref_cursor", None)
            if self.left_mode == "hold" and getattr(self, "_left_hold_joints", None) is not None \
                    and "left" in a:
                held = np.asarray(a["left"]["pos"], np.float32).copy()
                held[0:6] = self._left_hold_joints
                a["left"] = {"pos": held}                 # park the left arm during approach too
        elif mode == SubtaskMode.HUMAN and self.open_gripper_on_human:
            a = self._hold_with_open_gripper(obs)             # H18: free the empty gripper
        else:  # PAUSED / TERMINAL / HUMAN(no-open) -> hold (empty action)
            a = {}

        _t["mode_action"] = _time.perf_counter() - _t2
        self._debug_record(mode, obs, a, _t)

        a["_mode"] = self._mode_label(mode)
        a["_phase"] = a["_mode"]    # EpisodeRecorder phase.npy: rl/reset/human/vla per
                                    # frame (rl_collection.yaml) — without this the
                                    # DAgger recorder labels every frame 'autonomous'
        v = self._v
        a["_success"] = bool(v.get("success"))
        a["_reward"] = 1.0 if v.get("success") else 0.0
        a["_reset_source"] = ("scripted" if mode == SubtaskMode.RESET
                              else "human" if mode == SubtaskMode.HUMAN else "none")
        if self.collector is not None:
            # ONLY RL rollout steps go to the training collector (user rule 2026-07-08:
            # RL training needs RL rollouts only — reset/human/vla frames were journal
            # noise and frame-save overhead). The full-loop record with all phases
            # still lives in the rl_collection.yaml session recording (phase.npy).
            if mode == SubtaskMode.RL and not self._active_aux:
                self.collector.record_step(
                    obs, mode=mode.value, source="rl", reward=a["_reward"],
                    success=a["_success"], unrecoverable=bool(v.get("unrecoverable")),
                    reset_source=a["_reset_source"],
                    residual=self._last_residual,
                    right_action_source=a.get("_right_action_source", ""),
                    left_action_source=a.get("_left_action_source", ""))
            if self._end_episode_pending:
                # Deferred from _finalize_episode so the TERMINAL step (which carries the
                # success/unrecoverable flag) is recorded INSIDE the episode it closes.
                self._end_episode_pending = False
                if hasattr(self.collector, "end_episode"):
                    self.collector.end_episode()
        return a

    def action_spec(self):
        return self.vla.action_spec()

    @remote()
    def phase_log_label(self) -> str:
        """Label for launch.py's phase-edge log lines (instead of 'DAgger phase')."""
        return "SubRL loop"

    @remote()
    def phase_name(self) -> str:
        """Current loop mode for the TUI panel: 'rl' (RL rollout / pi0.5+residual driving),
        'reset' (coding-agent auto-reset), 'human' (Call-Human — scene needs restoring),
        'vla' / 'paused' / 'terminal'. Plumbed by launch.py -> tui.update_phase, same
        mechanism as the DAgger phase display."""
        return self._mode_label(self.machine.mode)

    def _mode_label(self, mode) -> str:
        """Phase string for logs/TUI/phase.npy: the RL mode is labelled with the
        ACTIVE SKILL (user 2026-08-03) — e.g. 'vla -> rl1_grasp' / 'vla -> rl2_insert'
        in the SubRL loop log instead of an ambiguous 'rl'."""
        if mode == SubtaskMode.RL:
            return self.aux_skill_label if self._active_aux else self.skill_label
        return mode.value

    def reset(self) -> None:
        self.machine.reset()
        self.reset_policy.reset()
        self._clear_chunk_state()
        self._prev_mode = None
        if hasattr(self.vla, "reset"):
            self.vla.reset()

    # -- mode-entry hooks (review M21/M23: state must be fresh on ENTRY) ------
    def _on_mode_entry(self, mode: SubtaskMode, obs) -> None:
        if mode == self._prev_mode:
            return
        prev = self._prev_mode
        self._prev_mode = mode
        if mode == SubtaskMode.RL:
            self._rl_seen = True     # the one-time VLA approach phase is over for good
            self._human_open_gripper = False   # the success-release is per-episode
            # fresh VLA chunk for the new scene (M21) + fresh RL chunk state + episode open
            if hasattr(self.vla, "reset"):
                self.vla.reset()
            self._clear_chunk_state()
            if self.left_mode == "hold" and self.hold_pose_left is None:
                # capture the CURRENT left joints as the park pose ('hold' previously fell
                # back to the VLA's left action — i.e. not a hold at all)
                try:
                    self._left_hold_joints = np.asarray(
                        obs["left"]["joint_pos"], np.float32).reshape(-1)[:6].copy()
                except Exception:
                    self._left_hold_joints = None
            if self.collector is not None and hasattr(self.collector, "start_episode") \
                    and not self._active_aux:
                self.collector.start_episode()             # aux segments are NOT episodes
        elif mode == SubtaskMode.VLA:
            if hasattr(self.vla, "reset"):
                self.vla.reset()                          # fresh chunk for the newly staged scene
            if self.left_mode == "hold" and self.hold_pose_left is None:
                try:
                    self._left_hold_joints = np.asarray(
                        obs["left"]["joint_pos"], np.float32).reshape(-1)[:6].copy()
                except Exception:
                    self._left_hold_joints = None
        elif mode == SubtaskMode.RESET:
            # NOTE: the gripper stays CLOSED here — after a success the vial is still HELD,
            # which is exactly the inverse (EAP) reset's precondition. Only HUMAN entry
            # (grasp wrecked / vial lost, gripper empty) opens the gripper.
            self._reset_retries = 0
            self._reset_attempt_start = self.machine.steps_in_mode
            self._arm_reset_policy()
        elif mode == SubtaskMode.HUMAN:
            self._human_ready = 0
            # FREEZE the arm pose for the whole HUMAN phase (user rule 2026-07-07:
            # the arm must keep the previous phase's last pose — echoing the observed
            # joints every tick let it drift/rotate under load, seen at frame 1890).
            try:
                self._human_hold_q = (
                    np.asarray(obs["left"]["joint_pos"], np.float32).reshape(-1)[:6].copy(),
                    np.asarray(obs["right"]["joint_pos"], np.float32).reshape(-1)[:6].copy())
            except Exception:
                self._human_hold_q = None
            # view-lift target: entry xy, z raised to human_lift_view_z (never down)
            self._human_lift_target = None
            self._human_q_cmd = None
            if self.human_lift_view_z > 0 and self.fk is not None and self._human_hold_q is not None:
                try:
                    pose = self.fk.ee_pose(self._human_hold_q[1])
                    if pose is not None and float(pose[6]) < self.human_lift_view_z:
                        self._human_lift_target = np.array(
                            [pose[4], pose[5], self.human_lift_view_z], np.float64)
                        self._human_q_cmd = self._human_hold_q[1].astype(np.float64).copy()
                        _log(f"HUMAN view lift: z {float(pose[6]):.3f} -> "
                             f"{self.human_lift_view_z:.3f} (wrist overlooks staging area)")
                except Exception:
                    self._human_lift_target = None
        _log(f"mode {getattr(prev, 'value', None)} -> {mode.value}")

    def _arm_reset_policy(self) -> None:
        """Fresh reset attempt: clear the phase machine (recaptures the carry orientation
        and resamples the random target) and plumb the LIVE grasp height (verifier: link_6
        z at the tick the gripper closed on the vial) in as the release height. Same FK
        frame, reachable by construction — the offline-calibrated z offsets sent the
        2026-07-06 21:24 reset to an unreachable target (IK saturated 596 ticks)."""
        self.reset_policy.reset()
        inner = getattr(self.reset_policy, "inner", self.reset_policy)
        gz = getattr(self.verifiers, "last_grasp_z", None)
        try:
            inner.place_z = float(gz) if gz is not None and np.isfinite(gz) else None
        except Exception:
            pass

    def _save_human_verify_frames(self, obs) -> str:
        """Dump the three camera frames at ESCALATION time (user rule 2026-07-07: the
        VLM sometimes misjudges fallen vs standing — the human verifies from the saved
        wrist image). Returns ' [frames: <dir>]' for the Call-Human reason, or ''."""
        try:
            import pathlib
            import time as _time

            from PIL import Image
            frames = []
            for cam in ("right_wrist_camera", "head_camera", "left_wrist_camera"):
                try:
                    frames.append((cam, np.asarray(obs[cam]["images"].get("rgb_hd", obs[cam]["images"]["rgb"]), np.uint8)))
                except Exception:
                    pass
            if not frames:            # camera-less obs (offline tests) -> no empty dirs
                return ""
            d = pathlib.Path("logs/human_verify") / _time.strftime("%Y%m%d_%H%M%S")
            d.mkdir(parents=True, exist_ok=True)
            for cam, rgb in frames:
                try:
                    Image.fromarray(rgb).save(d / f"{cam}.png")
                except Exception:
                    pass
            return f" [frames: {d}]"
        except Exception:
            return ""

    def _holding_vial(self, obs) -> bool:
        """Proprio holding check (width in the held band) — definitive while the vial is
        in the gripper, where every camera is blind to it."""
        try:
            w = float(np.asarray(obs["right"]["gripper_pos"]).reshape(-1)[0])
        except Exception:
            return False
        return 0.02 <= w <= 0.15

    # -- transitions -----------------------------------------------------------
    def _handle_transitions(self, obs) -> None:
        self.machine.consume_human()                      # human pedal first (override)
        in_aux = (self._active_aux and self.machine.mode == SubtaskMode.RL
                  and self.aux_verifiers is not None)
        v = (self.aux_verifiers if in_aux else self.verifiers).evaluate(obs)
        self._v = v
        mode = self.machine.mode

        if v.get("unrecoverable") and mode != SubtaskMode.HUMAN:
            if self.machine.profile == "inference" and not self.full_task_training:
                # NO HUMAN PHASE IN EVAL (user 2026-07-14): the operator at the
                # keyboard owns the scene during evaluation — a degraded sub-task
                # attempt just ends (reward 0) and hands back to the VLA; no
                # Call-Human, no resume gate. The operator labels the trial
                # (Space = failure) and restages.
                if mode == SubtaskMode.RL:
                    self._finalize_episode(obs, reward=0.0, cause="unrecoverable")
                    _log(f"unrecoverable during eval RL ({v.get('reason', '')}) -> "
                         "reward 0 -> back to VLA (no HUMAN phase at inference)")
                    self.machine.propose(SubtaskMode.VLA, force=True)
                return
            # NOT while already in HUMAN: the early return here starved the HUMAN
            # branch below of its resume gate AND its 15 s timeout — the verifier
            # keeps reporting unrecoverable until the human actually restores the
            # scene, so the loop sat in HUMAN for minutes (01:11 run: 128 s).
            if mode == SubtaskMode.RL:
                if self._active_aux:
                    _log(f"AUX skill segment unrecoverable ({v.get('reason', '')}) — "
                         "not collected")
                else:
                    self._finalize_episode(obs, reward=0.0, cause="unrecoverable")
            entered = self.machine.propose(SubtaskMode.HUMAN, force=True)
            if entered is not None:                       # just escalated -> Call Human ONCE
                self.escalator.call_human(
                    obs, reason=str(v.get("reason", "unrecoverable state"))
                    + self._save_human_verify_frames(obs))
            return

        # RL -> HUMAN rules (user final spec 2026-07-08): (1) verifier-labeled
        # SUCCESS (below) or verifier-CONFIRMED FALLEN, (2) rl_timeout. The old
        # agent-side 30-tick instantaneous vial_fallen streak is DELETED — SAM3
        # sometimes labels a STANDING vial fallen and one false verdict persists in
        # the ~2 s detection cache, killing rollouts before the policy reached the
        # vial (run 1932: ~30-step reward-0 episodes). The fallen->HUMAN handoff now
        # arrives via the verifier's `unrecoverable` (handled above): sustained
        # ~3 s of fallen AND a grasp attempted — a mislabel at rollout start can't
        # fire because the human guarantees a standing vial at RL entry.

        if mode == SubtaskMode.RL and v.get("success"):
            if self._active_aux:
                # AUX (frozen) skill segment done — e.g. grasp succeeded, vial HELD;
                # NOT an episode: nothing finalized/pushed. VLA carries on.
                _log("AUX skill SUCCESS (frozen policy) -> VLA continues the task")
                self.machine.propose(SubtaskMode.VLA, force=True)
                return
            # 0/1 verifier reward: this rollout SUCCEEDED (stable grasp+lift) -> reward 1,
            # close the episode, and hand off to the inverse reset (vial still held) —
            # or, in the HUMAN-RESET-ONLY mode, OPEN the gripper (release the vial for
            # the human) and hand the scene to the human.
            self._finalize_episode(obs, reward=1.0)
            # BATCH-SCENE counter (full-task insert training): after vials_per_scene
            # primary successes the table is empty and the stand full — hand the
            # WHOLE scene to the human for restaging (the only human touch per scene).
            self._scene_successes += 1
            if self.full_task_training and self.vials_per_scene > 0 \
                    and self._scene_successes >= self.vials_per_scene:
                entered = self.machine.propose(SubtaskMode.HUMAN, force=True)
                if entered is not None:
                    self.escalator.call_human(
                        obs, reason=f"scene exhausted ({self._scene_successes} vials "
                                    "inserted) — please restage ALL vials on the table")
                return
            if self.human_reset_only:
                self._human_open_gripper = True
                _log("RL SUCCESS (reward 1) -> gripper OPENS -> HUMAN reset "
                     "(human_reset_only mode)")
                self.machine.propose(SubtaskMode.HUMAN, force=True)
                return
            target = SubtaskMode.RESET if self.machine.profile == "training" else SubtaskMode.VLA
            self.machine.propose(target, force=True)
            return

        if mode == SubtaskMode.RL and v.get("failure"):
            # Verifier-labelled FAILURE (insert spec 2026-07-31: released the vial
            # but the stand count did not increase) — the episode ENDS with reward
            # 0 immediately instead of burning the rest of the RL budget.
            if self._active_aux:
                _log(f"AUX skill FAILURE ({v.get('reason', '')}) -> VLA continues")
                self.machine.propose(SubtaskMode.VLA, force=True)
                return
            self._finalize_episode(obs, reward=0.0, cause="insert_failed")
            if self.human_reset_only:
                _log(f"RL FAILURE ({v.get('reason', '')}) -> reward 0 -> HUMAN reset")
                self.machine.propose(SubtaskMode.HUMAN, force=True)
            else:
                _log(f"RL FAILURE ({v.get('reason', '')}) -> reward 0 -> back to VLA")
                self.machine.propose(SubtaskMode.VLA, force=True)
            return

        _rl_budget = self.aux_rl_timeout_steps if self._active_aux else self.rl_timeout_steps
        if mode == SubtaskMode.RL and self.machine.steps_in_mode > _rl_budget:
            if self._active_aux:
                _log(f"AUX skill segment timed out after {self.machine.steps_in_mode} "
                     "steps -> VLA continues (not collected)")
                self.machine.propose(SubtaskMode.VLA, force=True)
                return
            # recoverable failure: episode ends with reward 0; the reset re-stages the
            # scene (stage branch opens the gripper) and the cycle continues autonomously
            # — or, in the HUMAN-RESET-ONLY mode, the human re-stages.
            self._finalize_episode(obs, reward=0.0, cause="timeout")
            if self.human_reset_only:
                _log(f"RL episode timed out after {self.machine.steps_in_mode} steps -> "
                     "reward 0 -> HUMAN reset (human_reset_only mode)")
                self.machine.propose(SubtaskMode.HUMAN, force=True)
                return
            if self.machine.profile == "inference":
                # full-task eval: hand the arm BACK to the VLA to retry/continue —
                # there is no RL->RESET edge in the inference profile, so the old
                # RESET proposal was silently rejected and the agent re-finalized
                # the episode every tick (found preparing the 2026-07-13 eval).
                _log(f"RL sub-task timed out after {self.machine.steps_in_mode} steps -> "
                     "reward 0 -> back to VLA (inference)")
                self.machine.propose(SubtaskMode.VLA, force=True)
                return
            _log(f"RL episode timed out after {self.machine.steps_in_mode} steps -> reward 0, RESET")
            self.machine.propose(SubtaskMode.RESET, force=True)
            return

        if mode == SubtaskMode.RESET:
            # escape hatch (review H14): a stuck/failed reset escalates instead of wedging.
            # BUT (user rule 2026-07-06 22:35): while the vial is still safely HELD in the
            # gripper, a failed/stuck place-back is NOT a human matter — RETRY with a fresh
            # random target and a fresh orientation capture. Humans reset ONLY fallen/lost
            # vials.
            # SINGLE-ATTEMPT place-back (user rule 2026-07-07, matching the human
            # demos: the vial is placed ONCE — a failed/stuck attempt escalates, it
            # is never re-tried).
            if getattr(self.reset_policy, "failed", False) or \
                    self.machine.steps_in_mode > self.reset_timeout_steps:
                entered = self.machine.propose(SubtaskMode.HUMAN, force=True)
                if entered is not None:
                    self.escalator.call_human(
                        obs, reason="auto-reset failed or timed out — please re-stage the scene"
                                    + self._save_human_verify_frames(obs))
                return
            # INITIAL-STATE VERIFICATION before every RL entry (user rule 2026-07-06
            # 23:0x + 2026-07-07): the verifier must confirm the bottleneck sub-task's
            # start state with the THREE cameras — a STANDING vial visible (the VLM's
            # per-box pose; only a standing vial is a valid grasp target) and gripper
            # open — not just the reset's own done(). A fallen vial previously passed
            # done() (it is "visible") and RL re-entered pointlessly; now the loop
            # holds in RESET and the fallen/lost hysteresis escalates to Call-Human.
            start_verified = (v.get("start_ok", True) and v.get("vial_visible", True)
                              and v.get("vial_standing", True)
                              and not v.get("vial_fallen", False))
            if self.reset_policy.done(obs) and start_verified:
                # Scene staged + verified. VLA approach ONLY for the session's FIRST
                # cycle (boot); once RL has run, the reset ends with the gripper
                # retracted above the vial, so every later episode goes straight to RL.
                first_approach = self.use_vla_approach and not self._rl_seen
                self.machine.propose(SubtaskMode.VLA if first_approach else SubtaskMode.RL)
                return

        if mode == SubtaskMode.HUMAN:
            # Resume only when the scene is ACTUALLY restored: gripper open (start_ok)
            # AND the vial visible to the cameras AND standing upright (not lying on its
            # side) AND all sustained ~2 s. start_ok alone is true with the vial on the
            # floor and bounced out of HUMAN in <1 s (2026-07-06 21:25 run) — no human
            # had touched the scene yet.
            if bool(v.get("start_ok")) and bool(v.get("vial_visible", True)) \
                    and bool(v.get("vial_standing", True)) \
                    and not bool(v.get("vial_fallen", False)):
                self._human_ready += 1
            else:
                self._human_ready = 0
            timed_out = self.machine.steps_in_mode > self.human_timeout_steps
            if self._human_ready >= self.human_resume_frames or timed_out:
                if timed_out and self._human_ready < self.human_resume_frames:
                    _log(f"HUMAN phase timed out after {self.machine.steps_in_mode} steps "
                         "-> forcing RESET (scene re-verified before RL)")
                self._human_ready = 0
                # Fresh eyes on the human-restored scene (the user guarantees the
                # start state after a human reset): clear the verifier's latched
                # lost/fallen hysteresis so a stale pre-restore verdict can't
                # instantly re-escalate before the throttled detector catches up.
                if hasattr(self.verifiers, "_reset"):
                    self.verifiers._reset()
                if self.aux_verifiers is not None and hasattr(self.aux_verifiers, "_reset"):
                    self.aux_verifiers._reset()
                self._scene_successes = 0      # fresh scene after the batch restage
                self._active_aux = False
                # DIRECTLY to RL (user rule 2026-07-07): the human guarantees the
                # sub-task start state — no re-staging reset in between. With
                # human_resume_to="vla" (insert sub-task) the VLA takes over
                # instead and the selector re-fires the sub-task entry.
                if self.human_resume_to == "vla":
                    target = SubtaskMode.VLA
                else:
                    target = SubtaskMode.RL if self.machine.profile == "training" else SubtaskMode.VLA
                self.machine.propose(target, force=True)
            return

        if mode == SubtaskMode.VLA \
                and (self.machine.profile == "training" or self.full_task_training) \
                and self.machine.steps_in_mode > self.approach_timeout_steps:
            # Also active in the FULL-TASK loop (2026-07-31): failed inserts leave
            # fallen vials the grasp gate rightly refuses, so a scene can run out
            # of insertable vials BEFORE vials_per_scene successes — without this
            # the loop would idle in VLA forever instead of calling the human.
            entered = self.machine.propose(SubtaskMode.HUMAN, force=True)
            if entered is not None:
                self.escalator.call_human(
                    obs, reason="VLA ran with no sub-task entry for too long — scene "
                                "likely exhausted (fallen/finished vials); please restage"
                                + self._save_human_verify_frames(obs))
            return

        # AUX skill entry has PRIORITY in VLA (full-task loop): its entry state
        # (open gripper over a table vial) and the primary's (holding over the
        # stand) are mutually exclusive by proprio, so order is just determinism.
        if mode == SubtaskMode.VLA and self.aux_selector is not None:
            try:
                t = self.aux_selector.select(obs, mode.value)
            except Exception:
                t = None
            if t == "rl":
                self._active_aux = True
                self.machine.propose(SubtaskMode.RL)
                return
        sel = self.selector
        if self._active_aux and mode == SubtaskMode.RL and self.aux_selector is not None:
            sel = self.aux_selector                        # aux backstop exit (rl branch)
        tgt = sel.select(obs, mode.value)                  # coding-agent selector handoffs
        if tgt is not None:
            if mode == SubtaskMode.VLA:
                self._active_aux = False                   # primary skill entry
            try:
                self.machine.propose(SubtaskMode(tgt))
            except ValueError:
                pass

    # -- RL chunk machinery ------------------------------------------------------
    def _right_proprio(self, obs) -> np.ndarray:
        """RLT proprio for the right-arm sub-task: [joint_pos(6), gripper(1)] = 7."""
        try:
            r = obs["right"]
            jp = np.asarray(r["joint_pos"], np.float32).reshape(-1)[:6]
            gp = np.asarray(r.get("gripper_pos", [0.0]), np.float32).reshape(-1)[:1]
            return np.concatenate([jp, gp]).astype(np.float32)
        except Exception:
            if not self._warned_proprio:
                self._warned_proprio = True
                _log("right-arm proprio missing from obs — using zeros (replay proprio will "
                     "be meaningless until fixed)", warn=True)
            return np.zeros(7, np.float32)

    def _ref_slice(self, a_vla, C: int, adim: int) -> np.ndarray:
        """ALIGNED right-arm reference (review H10): slice the VLA chunk at its execution
        cursor so the residual is computed against the actions actually being executed."""
        ref_full = a_vla.get("_ref_chunk")
        if ref_full is None:
            return np.zeros((C, adim), np.float32)
        ref_full = np.asarray(ref_full, np.float32)
        k = int(a_vla.get("_ref_cursor") or 0)
        k = max(0, min(k, len(ref_full) - 1))
        sl = ref_full[k:k + C, 7:14] if ref_full.shape[-1] >= 14 else ref_full[k:k + C, :adim]
        if len(sl) < C:                                   # pad by repeating the last action
            sl = np.concatenate([sl, np.repeat(sl[-1:], C - len(sl), axis=0)], axis=0)
        return np.ascontiguousarray(sl, dtype=np.float32)

    def _rl_step(self, obs, a_vla) -> np.ndarray:
        """Chunk-level RL (review H9): query the actor once per C steps; execute the cached
        action chunk step-by-step as the residual; emit ONE transition per completed chunk.
        In the full-task loop the AUX (frozen) skill routes to its own actor service."""
        client = (self.aux_rl_client if (self._active_aux and self.aux_rl_client is not None)
                  else self.rl_client)
        if not hasattr(client, "infer_rlt"):              # legacy per-step client (tests/stubs)
            action = np.asarray(client.infer(obs)["actions"], np.float32)
            res = action[0] if action.ndim == 2 else action
            self._last_residual = res
            return res

        C = self.chunk_len
        adim = int(getattr(client, "action_dim", 7))
        # C=50 = pi0.5's full action horizon (user 2026-07-08): the RL boundary is
        # "a NEW pi0.5 chunk arrived" (async replan every ~1.2 s resets the exec
        # cursor), so the actor refines each VLA chunk exactly once, 1:1 — a fixed
        # C-tick timer would drift across replans and apply most of the refinement
        # against a reference the actor never saw. _chunk_pos >= C stays as the
        # safety boundary for a stalled replan.
        cur = int(a_vla.get("_ref_cursor") or 0)
        boundary = (self._cached_chunk is None or self._chunk_pos >= C
                    or cur < getattr(self, "_chunk_anchor", 0))
        if boundary:
            self._chunk_anchor = cur
            feats = {
                "z_rl": (None if a_vla.get("_z_rl") is None
                         else np.asarray(a_vla["_z_rl"], np.float32)),
                "proprio": self._right_proprio(obs),
                "ref_chunk": self._ref_slice(a_vla, C, adim),
            }
            chunk = np.asarray(
                client.infer_rlt(feats["z_rl"], feats["proprio"], feats["ref_chunk"]),
                np.float32)
            if chunk.ndim == 1:
                chunk = np.repeat(chunk[None, :], C, axis=0)
            # CHUNK-COHERENT execution (user rule 2026-07-08: NO per-step residual —
            # raw tanh rows have no temporal continuity, so executing them as
            # independent per-tick deltas injects +/-bound of fresh noise at 25 Hz
            # and a jump at every chunk boundary):
            #  (a) moving-average the rows along time (win 3) -> one smooth
            #      trajectory-level correction per chunk;
            #  (b) linearly blend the first rows from the previous chunk's last
            #      executed correction -> continuous across boundaries.
            # The SMOOTHED chunk is what executes AND what the transition stores
            # (action_chunk), so the learner trains on exactly what ran.
            sm = np.empty_like(chunk)
            for t in range(len(chunk)):
                sm[t] = chunk[max(0, t - 1):min(len(chunk), t + 2)].mean(axis=0)
            if self._last_residual is not None:
                prev = np.asarray(self._last_residual, np.float32).reshape(-1)
                if prev.shape == sm[0].shape:
                    B = min(3, len(sm))
                    for t in range(B):
                        w = (t + 1.0) / (B + 1.0)
                        sm[t] = (1.0 - w) * prev + w * sm[t]
            chunk = sm
            feats["action_chunk"] = chunk
            # the PREVIOUS chunk is now complete -> emit its transition (next_* = this boundary)
            self._emit_transition(next_feats=feats, done=False)
            self._chunk_feats = feats
            self._cached_chunk = chunk
            self._chunk_rewards = []
            self._chunk_pos = 0

        # Row selection ALIGNED to the VLA's execution cursor (row t of the refined
        # chunk corrects row t of the reference it was computed from), not a private
        # tick counter that would drift under async chunk blending.
        row = min(max(cur - getattr(self, "_chunk_anchor", 0), 0),
                  len(self._cached_chunk) - 1)
        res = self._cached_chunk[row]
        self._chunk_pos += 1
        self._chunk_rewards.append(0.0)                   # sparse; terminal stamps the last one
        self._last_residual = res
        return res

    def _emit_transition(self, next_feats: Dict[str, Any], done: bool) -> None:
        """Push the pending (executed) chunk as ONE RLT transition with rewards (C,)."""
        if self._chunk_feats is None or self.collector is None or self._active_aux:
            return                       # aux (frozen) skill chunks never enter the replay
        if not hasattr(self.collector, "record_rl_transition"):
            return
        C = self.chunk_len
        rew = np.zeros(C, np.float32)
        n = min(len(self._chunk_rewards), C)
        rew[:n] = np.asarray(self._chunk_rewards[:n], np.float32)
        self.collector.record_rl_transition(self._chunk_feats, rew, done, next_feats)
        self._chunk_feats = None

    def _finalize_episode(self, obs, reward: float, cause: str = "") -> None:
        """Terminal verdict while in RL (review C2): stamp the sparse reward onto the last
        executed step, flush the pending chunk with done=True, close the episode.
        `cause` labels WHY the episode ended (meta.json outcome for non-success/
        non-unrecoverable ends — 'timeout' vs the old catch-all 'incomplete',
        user 2026-07-08: 31-step false-fallen episodes were mislabeled incomplete)."""
        if self._chunk_rewards:
            self._chunk_rewards[-1] = float(reward)
        # Fresh verifier state for the NEXT episode regardless of outcome (replay
        # verify 2026-07-31): a TIMEOUT-ended insert episode left a PENDING release
        # candidate + stale confirm frames inside the verifier, which swallowed the
        # next episode's real release (the success path one-shot-resets itself, but
        # timeout/unrecoverable ends did not).
        if hasattr(self.verifiers, "_reset"):
            self.verifiers._reset()
        if self.collector is not None:
            if cause and hasattr(self.collector, "note_cause"):
                self.collector.note_cause(cause)
            # TERMINAL journal record: with RL-only recording (user 2026-07-08) the
            # mode has already flipped to RESET/HUMAN on this tick, so act()'s
            # RL-gated record_step would skip the very step that carries the episode
            # outcome — write it here explicitly, THEN stamp it terminal.
            self.collector.record_step(
                obs, mode="rl", source="rl", reward=float(reward),
                success=bool(reward > 0.0),
                unrecoverable=bool(self._v.get("unrecoverable")),
                reset_source="none", residual=self._last_residual,
                right_action_source="rl", left_action_source=self.left_mode)
            if hasattr(self.collector, "stamp_terminal"):
                self.collector.stamp_terminal(reward=reward, done=True)
            next_feats = {"z_rl": (self._chunk_feats or {}).get("z_rl"),
                          "proprio": self._right_proprio(obs),
                          "ref_chunk": (self._chunk_feats or {}).get("ref_chunk")}
            self._emit_transition(next_feats=next_feats, done=True)
            self._end_episode_pending = True   # closed AFTER this tick's record_step (see act)
        self._clear_chunk_state()

    # -- action composition ---------------------------------------------------
    def _compose(self, a_vla: Dict[str, Any], residual: np.ndarray) -> Dict[str, Any]:
        left = np.asarray(a_vla["left"]["pos"], np.float32).copy()
        right = np.asarray(a_vla["right"]["pos"], np.float32).copy()
        flat = np.concatenate([left, right])                # 14
        res = np.clip(np.asarray(residual, np.float32).reshape(-1), -1.0, 1.0)  # residual IS unit-bounded
        flat[self._ctrl] = flat[self._ctrl] + self.bound[self._ctrl] * res
        if self.action_low is not None and self.action_high is not None:        # joint limits (C5)
            flat = np.clip(flat, np.asarray(self.action_low, np.float32),
                           np.asarray(self.action_high, np.float32))
        if self.left_mode == "hold" and not any(i < 7 for i in self.controlled_indices):
            if self.hold_pose_left is not None:
                flat[0:7] = self.hold_pose_left
            elif getattr(self, "_left_hold_joints", None) is not None:
                flat[0:6] = self._left_hold_joints          # park joints; keep VLA's gripper cmd
        out = {"left": {"pos": flat[0:7]}, "right": {"pos": flat[7:14]}}
        out["_right_action_source"] = "rl" if any(i >= 7 for i in self.controlled_indices) else "vla"
        out["_left_action_source"] = ("rl" if any(i < 7 for i in self.controlled_indices)
                                      else self.left_mode)
        out["_controlled_indices"] = list(self.controlled_indices)
        return out

    def _hold_with_open_gripper(self, obs) -> Dict[str, Any]:
        """HUMAN mode after a wrecked grasp: hold the current pose but OPEN the right
        gripper (review H18: otherwise start_ok's open-width condition can never be met
        without the human physically backdriving the gripper).

        NEVER open while the gripper is still HOLDING the vial (width in the held band):
        on 2026-07-06 21:25 the reset timed out mid-place with the vial held, escalated,
        and THIS open dropped it from height. Held -> keep the grip closed; the human
        takes the vial out of the gripper themselves (width then leaves the band and the
        open command engages, restoring the H18 resume path)."""
        try:
            if self._human_hold_q is not None:             # FROZEN entry pose (no drift)
                lj, rj = self._human_hold_q
            else:
                lj = np.asarray(obs["left"]["joint_pos"], np.float32).reshape(-1)[:6]
                rj = np.asarray(obs["right"]["joint_pos"], np.float32).reshape(-1)[:6]
            # VIEW LIFT (user 2026-07-27): one bounded DLS step per tick toward the
            # raised target, integrating from the last COMMANDED joints (stiction
            # rule); once within tol the lifted pose becomes the frozen hold.
            if getattr(self, "_human_lift_target", None) is not None and                     getattr(self, "_human_q_cmd", None) is not None:
                try:
                    q_obs = np.asarray(obs["right"]["joint_pos"], np.float64).reshape(-1)[:6]
                    pose = self.fk.ee_pose(self._human_q_cmd)
                    err = self._human_lift_target - np.asarray(pose[4:7], np.float64)
                    if float(np.linalg.norm(err)) < 0.01:
                        self._human_hold_q = (lj, self._human_q_cmd.astype(np.float32).copy())
                        self._human_lift_target = None      # reached: freeze lifted pose
                        rj = self._human_hold_q[1]
                    else:
                        J = self.fk.position_jacobian(self._human_q_cmd)
                        if J is None:
                            self._human_lift_target = None
                        else:
                            lam = 0.05
                            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3),
                                                       np.clip(err, -0.03, 0.03))
                            dq = np.clip(dq, -self.human_lift_step_rad, self.human_lift_step_rad)
                            q_new = self._human_q_cmd + dq
                            q_new = np.clip(q_new, q_obs - self.human_lift_lead_rad,
                                            q_obs + self.human_lift_lead_rad)
                            self._human_q_cmd = q_new
                            rj = q_new.astype(np.float32)
                except Exception:
                    self._human_lift_target = None          # any failure -> frozen hold
            lg = np.asarray(obs["left"].get("gripper_pos", [0.0]), np.float32).reshape(-1)[:1]
            width = float(np.asarray(obs["right"]["gripper_pos"]).reshape(-1)[0])
            holding = 0.02 <= width <= 0.15                # held band, observation units
            # human_reset_only SUCCESS: open IMMEDIATELY (user 2026-07-08: release the
            # lifted vial for the human to re-stage) — overrides the keep-held rule.
            grip = self.gripper_open_cmd if self._human_open_gripper \
                else (0.0 if holding else self.gripper_open_cmd)
            return {"left": {"pos": np.concatenate([lj, lg])},
                    "right": {"pos": np.concatenate([rj, [np.float32(grip)]])}}
        except Exception:
            return {}                                      # malformed obs -> plain hold

    # -- flight recorder ----------------------------------------------------------
    def _debug_record(self, mode, obs, a, timings) -> None:
        self._dbg_n = getattr(self, "_dbg_n", 0) + 1
        if self._dbg_n % max(1, self.debug_every):
            return
        try:
            import json
            import pathlib
            import time as _time
            rec = {"t": round(_time.time(), 2), "mode": mode.value,
                   "steps_in_mode": self.machine.steps_in_mode}
            rec.update({f"{k}_ms": round(v * 1000, 1) for k, v in timings.items()})
            try:
                cmd = np.asarray(a["right"]["pos"][:6], np.float32)
                cur = np.asarray(obs["right"]["joint_pos"], np.float32).reshape(-1)[:6]
                rec["right_delta_rad"] = round(float(np.max(np.abs(cmd - cur))), 4)
                rec["right_grip_cmd"] = round(float(a["right"]["pos"][6]), 3)
            except Exception:
                rec["right_delta_rad"] = None      # None = NO right command left the agent
            try:                                    # VLA chunk-buffer state (async health)
                rec["chunk_remaining"] = int(self.vla._chunk_mgr.remaining)
            except Exception:
                pass
            try:                                    # descent geometry (2026-07-06 unreachable-z grind)
                rec["ee_z"] = round(float(obs["right"]["ee_pose"][6]), 4)
                rec["ee_xy"] = [round(float(obs["right"]["ee_pose"][4]), 3),
                                round(float(obs["right"]["ee_pose"][5]), 3)]
                rec["grip_w"] = round(float(np.asarray(obs["right"]["gripper_pos"]).reshape(-1)[0]), 3)
            except Exception:
                pass
            if mode == SubtaskMode.RESET:
                try:
                    inner = getattr(self.reset_policy, "inner", self.reset_policy)
                    rec["reset_phase"] = getattr(inner, "phase", None)
                    txy = getattr(inner, "_target_xy", None)
                    rec["reset_target"] = None if txy is None else [round(float(txy[0]), 3),
                                                                    round(float(txy[1]), 3)]
                    pz = getattr(inner, "place_z", None)
                    rec["place_z"] = None if pz is None else round(float(pz), 4)
                    ctl = getattr(self.reset_policy, "control", None)
                    if ctl is not None:
                        rec["ori_err"] = round(float(ctl.ori_error()), 3)
                        rec["tilt_err"] = round(float(ctl.tilt_error()), 3)
                except Exception:
                    pass
            v = self._v
            rec["start_ok"] = bool(v.get("start_ok")); rec["success"] = bool(v.get("success"))
            rec["unrecoverable"] = bool(v.get("unrecoverable"))
            if v.get("vial_fallen"):
                rec["fallen"] = True
            if not v.get("vial_standing", True):
                rec["no_standing"] = True
            if v.get("reason"):
                rec["reason"] = str(v.get("reason"))[:220]   # long enough to keep the
                                                             # human-verify frames path
            try:                                    # live detector verdicts per camera
                from .perception import _SHARED
                for det in _SHARED.values():
                    with det._lock:
                        for cam, r in det._cache.items():
                            rec[f"det_{cam.split('_')[0]}"] = (None if r is None
                                                               else bool(r.get("found")))
            except Exception:
                pass
            path = pathlib.Path(self.debug_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    # -- startup validation (review C1, partial) --------------------------------
    def _validate_obs_once(self, obs) -> None:
        if self._validated or not obs:
            return
        self._validated = True
        missing = []
        for side, key in _REQUIRED_OBS_KEYS:
            try:
                _ = obs[side][key]
            except Exception:
                missing.append(f"{side}.{key}")
        if missing:
            _log("obs is MISSING keys the verifier/reset require: " + ", ".join(missing) +
                 " — the loop will not progress until these are populated (ee_pose needs FK "
                 "injection in this launch path; joint_eff needs the driver to report efforts)",
                 warn=True)
