"""RL policy client — implements limb's PolicyClient protocol for the sub-task RL
actor (the residual the SubtaskRLAgent adds to the frozen VLA action).

Modes:
- "dummy": returns a zero residual (identity over the VLA action). For off-robot
  plumbing validation — confirms composition + mode machine + logging work before
  any learning.
- "http":  POST the obs to the openpi-RLT actor service (:9101) and read the
  refined residual chunk back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RLPolicyClient:
    mode: str = "dummy"                 # "dummy" | "http"
    action_dim: int = 7                # 7 (single-arm residual) | 14 (dual-arm)
    action_horizon: int = 1
    url: str = "http://127.0.0.1:9101/infer"
    timeout_s: float = 1.0
    # FAILURE BACKOFF (2026-08-03, session 213025: a hung/down actor cost the 1 s
    # timeout on EVERY tick — the failed chunk makes the agent retry per tick — and
    # dragged the 30 Hz loop to 4 Hz for whole rl1_grasp segments): after a failed
    # call, return the zero residual IMMEDIATELY for this many seconds before the
    # next network attempt.
    fail_backoff_s: float = 2.0
    # EVAL (inference profile): True = the actor serves its mean action, no
    # exploration noise. Training keeps False (fixed-std Gaussian on the residual).
    deterministic: bool = False

    def __post_init__(self) -> None:
        self._connected = False
        self._last_warn = 0.0
        self._fail_until = 0.0

    def connect(self) -> None:
        self._connected = True

    def _warn(self, msg: str) -> None:
        """Rate-limited (5 s) warning so a down actor doesn't spam the 30 Hz loop."""
        import time
        now = time.monotonic()
        if now - self._last_warn >= 5.0:
            self._last_warn = now
            try:
                from loguru import logger
                logger.warning(msg)
            except Exception:
                pass

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        zero = {"actions": np.zeros((self.action_horizon, self.action_dim), np.float32)}
        if self.mode == "dummy":
            return zero
        # http: send obs to the RLT actor, expect {"actions": [[...]] } (residual in [-1,1])
        try:
            import requests
            r = requests.post(self.url, json={"obs": _to_jsonable(obs)}, timeout=self.timeout_s)
            r.raise_for_status()
            a = np.asarray(r.json()["actions"], np.float32)
        except Exception as e:                          # actor down != robot dead (review H12)
            self._warn(f"RL actor unreachable ({e}); falling back to zero residual (pure VLA)")
            return zero
        if a.ndim == 1:
            a = a[None, :]
        return {"actions": a}

    def infer_rlt(self, z_rl, proprio, ref_chunk, *, episode_id: int = 0,
                  deterministic: Optional[bool] = None) -> np.ndarray:
        """openpi-RLT actor contract, verified from its code: msgpack_numpy POST to
        /infer with ActorRequest{z_rl, proprio, ref_chunk, request_id, episode_id,
        step_id, deterministic, timestamp}; the response's action chunk is under
        'refined_chunk'. On ANY failure returns the ZERO chunk (pure frozen-VLA
        behavior) with a rate-limited warning (review H12) — the control loop must
        never die on an actor outage."""
        C = int(np.asarray(ref_chunk).shape[0]) if ref_chunk is not None else self.action_horizon
        zero = np.zeros((C, self.action_dim), np.float32)
        if self.mode == "dummy":
            return zero
        import time as _t
        if _t.monotonic() < self._fail_until:      # backoff window: no network attempt
            return zero
        try:
            import time
            import uuid

            import requests

            from limb.vendor.openpi_client import msgpack_numpy
            self._step_id = getattr(self, "_step_id", 0) + 1
            payload = {
                "z_rl": (np.zeros(0, np.float32) if z_rl is None
                         else np.asarray(z_rl, np.float32).reshape(-1)),
                "proprio": np.asarray(proprio, np.float32).reshape(-1),
                "ref_chunk": np.asarray(ref_chunk, np.float32),
                "request_id": uuid.uuid4().hex,
                "episode_id": int(episode_id),
                "step_id": int(self._step_id),
                "deterministic": bool(self.deterministic if deterministic is None
                                      else deterministic),
                "timestamp": time.time(),
            }
            r = requests.post(self.url, data=msgpack_numpy.Packer().pack(payload),
                              headers={"Content-Type": "application/octet-stream"},
                              timeout=self.timeout_s)
            r.raise_for_status()
            resp = msgpack_numpy.unpackb(r.content)
            a = np.asarray(resp["refined_chunk"], np.float32)
            return a[None, :] if a.ndim == 1 else a
        except Exception as e:
            self._fail_until = _t.monotonic() + self.fail_backoff_s
            self._warn(f"RLT actor unreachable ({e}); zero action chunk (pure VLA), "
                       f"backing off {self.fail_backoff_s:.1f}s")
            return zero

    def get_metadata(self) -> Dict[str, Any]:
        return {"action_dim": self.action_dim, "action_horizon": self.action_horizon,
                "residual": True, "mode": self.mode}

    def close(self) -> None:
        self._connected = False


def _to_jsonable(obs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _to_jsonable(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out
