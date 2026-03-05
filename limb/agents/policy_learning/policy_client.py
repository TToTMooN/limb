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
        from openpi_client import websocket_client_policy as wcp

        self._client = wcp.WebsocketClientPolicy(host=self.host, port=self.port)
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
    """

    host: str = "0.0.0.0"
    port: int = 8000
    connect_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        self._ws: Any = None
        self._metadata: Optional[Dict[str, Any]] = None

    def connect(self) -> None:
        import msgpack
        import msgpack_numpy
        from websockets.sync.client import connect

        msgpack_numpy.patch()

        uri = f"ws://{self.host}:{self.port}"
        self._ws = connect(uri, open_timeout=self.connect_timeout_s)
        # Server sends metadata immediately on connect
        raw = self._ws.recv()
        self._metadata = msgpack.unpackb(raw, raw=False)
        logger.info("Connected to policy server at {}. Metadata: {}", uri, self._metadata)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        import msgpack

        if self._ws is None:
            self.connect()
        packed = msgpack.packb(obs, use_bin_type=True)
        self._ws.send(packed)
        raw = self._ws.recv()
        response = msgpack.unpackb(raw, raw=False)
        if "error" in response:
            raise RuntimeError(f"Policy server error: {response['error']}")
        return response

    def get_metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            self.connect()
        return self._metadata

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None
