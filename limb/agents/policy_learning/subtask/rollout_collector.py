"""RolloutCollector — logs the online RL + auto-reset loop's data.

Two products per episode:
- **journal** (human-readable, per control step): mode/source/reward/success/residual +
  downsampled 3-camera frames — the coding-agent data (CaP-X-style) + the audit trail.
- **RLT chunk transitions** (the learner feed): one transition per RL action chunk,
  ``{z_rl, proprio, ref_chunk, action_chunk, rewards:(C,), done, next_z_rl,
  next_proprio, next_ref_chunk}`` — matching openpi-RLT's replay schema. Assembled by
  ``SubtaskRLAgent`` (which owns chunk boundaries) and handed over via
  ``record_rl_transition``; pushed to the replay service on ``end_episode``.

Episode lifecycle (review C3): ``SubtaskRLAgent`` calls ``start_episode()`` on entry to
RL and ``end_episode()`` on the terminal verdict. ``record_step`` also self-heals by
auto-starting an episode if none is open, and all mkdirs use ``parents=True``.

Terminal stamping (review C2): ``stamp_terminal`` back-writes the sparse reward/done
onto the LAST recorded RL step in the journal, since the success verdict is only
observable on the tick *after* the action that caused it.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

CAMS = ["head_camera", "left_wrist_camera", "right_wrist_camera"]


def proprio_state(obs: Dict[str, Any]) -> np.ndarray:
    """[left joint_pos(6), left gripper(1), right joint_pos(6), right gripper(1)] = 14."""
    def arm(side):
        a = obs[side]
        jp = np.asarray(a["joint_pos"], np.float32).reshape(-1)[:6]
        gp = np.asarray(a.get("gripper_pos", [0.0]), np.float32).reshape(-1)[:1]
        return np.concatenate([jp, gp])
    try:
        return np.concatenate([arm("left"), arm("right")]).astype(np.float32)
    except Exception:
        return np.zeros(14, np.float32)


class RolloutCollector:
    def __init__(self, out_dir: str = "recordings/subtask_rl", push_url: Optional[str] = None,
                 save_frames_every: int = 6,
                 episode_log_url: Optional[str] = "http://127.0.0.1:9101/episode"):
        # Per-SESSION subfolder stamped at launch time (user 2026-07-08): the ep0000/
        # ep0001/... dirs of different runs used to pile into one flat folder, so an
        # epNNNN couldn't be traced back to its full recording. The timestamp matches
        # the launch moment — i.e. the subrl_online_rl_..._<same timestamp> recording
        # session started seconds apart in the same process.
        from datetime import datetime
        session = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out = pathlib.Path(out_dir) / session
        self.out.mkdir(parents=True, exist_ok=True)
        self.push_url = push_url
        self.episode_log_url = episode_log_url   # RL service /episode -> wandb episode_reward
        self.save_frames_every = save_frames_every
        self._ep = -1
        self._open = False
        self._steps: List[dict] = []
        self._trans: List[dict] = []          # RLT chunk transitions for this episode
        self._t = 0
        self.pushed_total = 0                  # lifetime transitions accepted by the replay

    # -- episode lifecycle ----------------------------------------------------
    def start_episode(self, ep_id: Optional[int] = None) -> None:
        self._ep = ep_id if ep_id is not None else self._ep + 1
        self._steps = []
        self._trans = []
        self._t = 0
        self._open = True
        self._cause = ""                       # why the episode ended (note_cause)
        (self.out / f"ep{self._ep:04d}").mkdir(parents=True, exist_ok=True)

    def note_cause(self, cause: str) -> None:
        """Label WHY the closing episode ended ('timeout', 'unrecoverable', ...) —
        meta.json's outcome for non-success ends; 'incomplete' then only means the
        session was shut down mid-episode (user 2026-07-08)."""
        self._cause = str(cause)

    def record_step(self, obs: Dict[str, Any], *, mode: str, source: str, reward: float,
                    success: bool, unrecoverable: bool, reset_source: str,
                    residual: Optional[np.ndarray] = None,
                    right_action_source: str = "", left_action_source: str = "") -> None:
        if not self._open:                     # self-heal: never crash the control loop (C3)
            self.start_episode()
        s = proprio_state(obs)
        res = None if residual is None else np.asarray(residual, np.float32)
        rec = {
            "t": self._t, "mode": mode, "source": source, "reward": float(reward),
            "success": bool(success), "unrecoverable": bool(unrecoverable),
            "reset_source": reset_source, "right_action_source": right_action_source,
            "left_action_source": left_action_source,
            "proprio": s.tolist(),
            "residual": None if res is None else res.tolist(),
            # Honest sub-action-space audit (review M26: the old posture-norm 'ratio' was
            # meaningless): the RL influence is exactly the residual, so log its norm; a
            # nonzero residual with left_action_source != 'rl' proves right-arm-only.
            "residual_norm": None if res is None else float(np.linalg.norm(res)),
        }
        self._steps.append(rec)
        if self.save_frames_every and (self._t % self.save_frames_every == 0):
            self._save_frames(obs)
        self._t += 1

    def stamp_terminal(self, reward: float, done: bool = True) -> None:
        """Back-write the terminal verdict onto the last recorded RL step (review C2:
        success is detected on the tick AFTER the action that achieved it, by which time
        the mode has already flipped — without this the sparse reward is lost)."""
        for rec in reversed(self._steps):
            if rec["mode"] == "rl":
                rec["reward"] = float(reward)
                rec["success"] = bool(reward >= 0.5)
                rec["terminal"] = bool(done)
                return

    # -- RLT learner feed -------------------------------------------------------
    def record_rl_transition(self, feats: Dict[str, Any], rewards: np.ndarray, done: bool,
                             next_feats: Dict[str, Any]) -> None:
        """One CHUNK-level transition in openpi-RLT's exact RLTTransition schema (verified
        from its replay.py from_mapping): required keys incl. `source` (RL=1); rewards (C,)."""
        def _a(x, dt=np.float32):
            return np.zeros(0, dt) if x is None else np.asarray(x, dt)
        rew = _a(rewards)
        self._trans.append({
            "z_rl": _a(feats.get("z_rl")), "proprio": _a(feats.get("proprio")),
            "ref_chunk": _a(feats.get("ref_chunk")), "action_chunk": _a(feats.get("action_chunk")),
            "rewards": rew, "done": bool(done),
            "next_z_rl": _a(next_feats.get("z_rl")), "next_proprio": _a(next_feats.get("proprio")),
            "next_ref_chunk": _a(next_feats.get("ref_chunk")),
            "source": 1,                                   # TransitionSource.RL
            "collection_phase": "online_rl",
            "success": int(bool(done) and float(rew.max(initial=0.0)) > 0.0),
            "intervention_flag": False,
            "episode_id": int(self._ep), "step_id": len(self._trans),
        })

    def end_episode(self) -> Optional[pathlib.Path]:
        if not self._open:
            return None
        self._open = False
        ep_dir = self.out / f"ep{self._ep:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "rl_journal.jsonl").write_text("\n".join(json.dumps(r) for r in self._steps))
        outcome = ("success" if any(r.get("success") for r in self._steps)
                   else "unrecoverable" if any(r.get("unrecoverable") for r in self._steps)
                   else getattr(self, "_cause", "") or "incomplete")
        n_reward = sum(1 for tr in self._trans
                       if np.asarray(tr.get("rewards", ()), np.float32).max(initial=0.0) > 0.0)
        meta = {"episode": self._ep, "steps": len(self._steps), "outcome": outcome,
                "rl_transitions": len(self._trans), "rewarded_transitions": n_reward,
                "ts": time.time()}
        (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self._log(f"episode {self._ep}: outcome={outcome} steps={len(self._steps)} "
                  f"rl_transitions={len(self._trans)} rewarded={n_reward}")
        if self.push_url and self._trans:
            self._push()
        if self.episode_log_url:
            # Per-episode outcome to the RL service -> wandb episode_reward /
            # success-rate curves (user 2026-07-08). Fire-and-forget: a down RL
            # service must never affect the control loop.
            try:
                import requests
                requests.post(self.episode_log_url,
                              json={"episode": self._ep,
                                    "reward": 1.0 if outcome == "success" else 0.0,
                                    "outcome": outcome, "steps": len(self._steps),
                                    "rl_transitions": len(self._trans),
                                    "rewarded_transitions": n_reward},
                              timeout=2.0)
            except Exception as e:
                self._log(f"episode-log POST failed ({e}); wandb episode_reward misses ep{self._ep}",
                          warn=True)
        return ep_dir

    # -- internals ---------------------------------------------------------------
    def _log(self, msg: str, warn: bool = False) -> None:
        try:
            from loguru import logger
            (logger.warning if warn else logger.info)("[RolloutCollector] " + msg)
        except Exception:
            pass

    def _save_frames(self, obs: Dict[str, Any]) -> None:
        try:
            import imageio
        except Exception:
            return
        try:
            d = self.out / f"ep{self._ep:04d}" / "frames"
            d.mkdir(parents=True, exist_ok=True)          # parents=True (review C3)
            for cam in CAMS:
                if cam in obs and "images" in obs[cam] and "rgb" in obs[cam]["images"]:
                    imageio.imwrite(d / f"{cam}_{self._t:05d}.png",
                                    np.asarray(obs[cam]["images"]["rgb"]))
        except Exception as e:
            self._log(f"frame save failed at t={self._t}: {e}", warn=True)

    def _push(self) -> None:
        """Push this episode's chunk transitions to the openpi-RLT replay via
        msgpack_numpy POST /extend {'transitions': [...]} (its verified wire format).
        LOUD on failure (review H11: silent pass = hours of training on an empty replay)."""
        try:
            import requests

            from limb.vendor.openpi_client import msgpack_numpy
            r = requests.post(self.push_url,
                              data=msgpack_numpy.Packer().pack({"transitions": self._trans}),
                              headers={"Content-Type": "application/octet-stream"}, timeout=5.0)
            r.raise_for_status()
            self.pushed_total += len(self._trans)
            self._log(f"pushed {len(self._trans)} transitions to replay "
                      f"(lifetime {self.pushed_total})")
        except Exception as e:
            self._log(f"REPLAY PUSH FAILED ({len(self._trans)} transitions LOST): {e}", warn=True)
