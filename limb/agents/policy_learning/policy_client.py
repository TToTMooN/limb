"""Policy client protocol and implementations for communicating with external policy servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from loguru import logger


class PolicyClient(Protocol):
    """Protocol for communicating with an external policy inference server.

    A PolicyClient is a stateless transport layer — sends observations,
    receives action chunks. No buffering or smoothing.
    """

    def connect(self) -> None: ...

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Send observation, receive response with at least {"actions": ndarray}."""
        ...

    def get_metadata(self) -> Dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass
class OpenPIClient:
    """Client for OpenPI's native WebSocket protocol (pi0, pi0-FAST, pi0.5).

    Wraps openpi_client directly. Use when connecting to a server running
    OpenPI's own ``serve_policy.py``.
    """

    host: str = "0.0.0.0"
    port: int = 8111

    def __post_init__(self) -> None:
        self._client: Any = None

    def connect(self) -> None:
        # Uses the minimal vendored copy of openpi-client (see
        # limb/vendor/openpi_client/NOTICE.md). No external dep needed.
        from limb.vendor.openpi_client import WebsocketClientPolicy

        self._client = WebsocketClientPolicy(host=self.host, port=self.port)
        logger.info("OpenPIClient connected to {}:{}", self.host, self.port)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is None:
            self.connect()
        return self._client.infer(obs)

    def get_metadata(self) -> Dict[str, Any]:
        if self._client is None:
            self.connect()
        return self._client.get_server_metadata()

    def close(self) -> None:
        self._client = None


@dataclass
class WebSocketPolicyClient:
    """Generic WebSocket + msgpack client matching docs/policy_server_spec.md.

    Use when connecting to a policy server that implements the standard
    limb policy server protocol (msgpack serialization, metadata on connect).

    Auto-reconnects on dropped connections — needed because long-running
    inference loops will outlive transient network drops, server restarts,
    and idle-timeout disconnects.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    connect_timeout_s: float = 10.0
    max_reconnect_attempts: int = 5
    reconnect_backoff_s: float = 1.0

    def __post_init__(self) -> None:
        self._ws: Any = None
        self._metadata: Optional[Dict[str, Any]] = None
        # Use explicit numpy hooks instead of monkey-patching the msgpack module —
        # patch() has process-wide side effects that affect any other code importing msgpack.
        import msgpack_numpy

        self._encode = msgpack_numpy.encode
        self._decode = msgpack_numpy.decode

    def connect(self) -> None:
        import msgpack
        from websockets.sync.client import connect

        uri = f"ws://{self.host}:{self.port}"
        self._ws = connect(uri, open_timeout=self.connect_timeout_s, max_size=None)
        # Server sends metadata immediately on connect
        raw = self._ws.recv()
        self._metadata = msgpack.unpackb(raw, raw=False, object_hook=self._decode)
        logger.info("Connected to policy server at {}. Metadata: {}", uri, self._metadata)

    def _reconnect(self) -> None:
        """Tear down a dead connection and re-establish, with bounded retries."""
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        self._ws = None

        import time

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_reconnect_attempts + 1):
            try:
                self.connect()
                logger.info("Reconnected to policy server (attempt {}/{}).", attempt, self.max_reconnect_attempts)
                return
            except Exception as e:
                last_exc = e
                wait_s = self.reconnect_backoff_s * attempt
                logger.warning(
                    "Reconnect attempt {}/{} failed: {}. Retrying in {:.1f}s.",
                    attempt, self.max_reconnect_attempts, e, wait_s,
                )
                time.sleep(wait_s)
        raise ConnectionError(f"Failed to reconnect after {self.max_reconnect_attempts} attempts: {last_exc}")

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        import msgpack
        from websockets.exceptions import ConnectionClosed

        if self._ws is None:
            self.connect()
        packed = msgpack.packb(obs, use_bin_type=True, default=self._encode)
        try:
            self._ws.send(packed)
            raw = self._ws.recv()
        except (ConnectionClosed, ConnectionError, OSError) as e:
            logger.warning("Connection to policy server dropped ({}); reconnecting.", e)
            self._reconnect()
            self._ws.send(packed)
            raw = self._ws.recv()

        # Server may send an error frame as a string (OpenPI convention) or as
        # a msgpack dict with an "error" key (limb spec). Handle both.
        if isinstance(raw, str):
            raise RuntimeError(f"Policy server error: {raw}")
        response = msgpack.unpackb(raw, raw=False, object_hook=self._decode)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Policy server error: {response['error']}")
        return response

    def get_metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            self.connect()
        return self._metadata

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
