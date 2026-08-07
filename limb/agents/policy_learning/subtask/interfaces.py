"""Protocols + runnable defaults for the sub-task agent's pluggable parts.

The coding-agent harness authors the real Selector / ResetPolicy / Verifiers
(perception code). These defaults let the agent + mode machine + logging run and
be unit-tested off-robot before those are authored.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol


class SubtaskVerifiers(Protocol):
    def evaluate(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return {'start_ok': bool, 'success': bool, 'unrecoverable': bool, 'reason': str?}.

        RoboClaw failure taxonomy:
          - non-degrading (recoverable): state ~unchanged, retry same policy (missed grasp,
            vial still upright) -> stay in RL / EAP reset; 'unrecoverable' stays False.
          - degrading (unrecoverable): state altered, cannot retry (vial knocked over / on the
            floor / out of preconditions) -> 'unrecoverable'=True + a human-readable 'reason'.
        """
        ...


class HumanEscalator(Protocol):
    """RoboClaw 'Call Human' tool: summon a human when the state is unrecoverable / a safety
    limit is hit. Real impls notify via MCP / pedal-light / UI and (optionally) block until the
    human restores the scene (verifier 'start_ok' then resumes the loop)."""
    def call_human(self, obs: Dict[str, Any], reason: str) -> None: ...


class ResetPolicy(Protocol):
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]: ...
    def done(self, obs: Dict[str, Any]) -> bool: ...
    def reset(self) -> None: ...


class Selector(Protocol):
    def select(self, obs: Dict[str, Any], mode: str) -> Optional[str]:
        """Return a target mode name to switch to, or None to stay."""
        ...


# ---- runnable defaults ----------------------------------------------------

class CallableVerifiers:
    """Wrap plain predicates (e.g. authored later by the coding agent)."""
    def __init__(self, start_ok: Callable | None = None, success: Callable | None = None,
                 unrecoverable: Callable | None = None):
        self._s, self._ok, self._u = start_ok, success, unrecoverable

    def evaluate(self, obs):
        f = lambda fn: bool(fn(obs)) if fn is not None else False
        return {"start_ok": f(self._s), "success": f(self._ok), "unrecoverable": f(self._u)}


class HoldReset:
    """Trivial reset: hold for ``settle`` ticks then report done. Placeholder for
    the coding-agent-authored END->START reset routine."""
    def __init__(self, settle: int = 10):
        self.settle = settle
        self._t = 0

    def act(self, obs):
        self._t += 1
        return {}                      # empty action -> robots hold (PD)

    def done(self, obs):
        return self._t >= self.settle

    def reset(self):
        self._t = 0


class NullSelector:
    """No selector-driven switches (verifiers drive everything). The
    coding-agent-authored selector replaces this for inference-time VLA<->RL."""
    def select(self, obs, mode):
        return None


class LoggingEscalator:
    """Default 'Call Human': log the request (loguru) + fire an optional callback (e.g. a
    pedal-light, a Slack/MCP notification, or a UI banner). Non-blocking — the mode machine
    already sits in HUMAN until the verifier reports 'start_ok' (human restored the scene)."""
    def __init__(self, notify: Callable[[str], None] | None = None):
        self._notify = notify

    def call_human(self, obs, reason: str) -> None:
        msg = f"[CALL HUMAN] unrecoverable state — please restore the scene. reason: {reason}"
        try:
            from loguru import logger
            logger.warning(msg)
        except Exception:
            print(msg)
        if self._notify is not None:
            try:
                self._notify(msg)
            except Exception:
                pass


class ScriptedVerifiers:
    """Time-scripted verifier for an autonomous on-robot DRY-RUN (no perception):
    fires `success` once every `period` evaluate() calls so the RESET<->RL<->RESET
    loop cycles on its own. Replace with the coding-agent-authored perception
    verifiers (which read the 3-camera + state obs) for real grasp detection."""
    def __init__(self, period: int = 60, start_ok: bool = True):
        self.period = period; self.start_ok = start_ok; self._n = 0
    def evaluate(self, obs):
        self._n += 1
        return {"start_ok": self.start_ok, "success": self._n % self.period == 0,
                "unrecoverable": False}
