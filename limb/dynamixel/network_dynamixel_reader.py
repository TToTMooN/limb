"""
Network-based Dynamixel position reader.

Connects to a ``gello_position_server`` running on the R1 Lite Teleop device
and receives joint positions over TCP.  Provides the same public interface as
:class:`DynamixelReader` so the two are interchangeable in :class:`YamGelloAgent`.
"""

import logging
import socket
import struct
import time
from threading import Event, Lock, Thread

import numpy as np

logger = logging.getLogger(__name__)


class NetworkDynamixelReader:
    """Receive joint positions from a remote ``gello_position_server`` over TCP.

    Parameters
    ----------
    host : str
        IP address of the R1 Lite Teleop (e.g. ``"10.42.0.1"``).
    port : int
        TCP port the server is listening on.
    reconnect_interval : float
        Seconds to wait before retrying after a connection drop.
    """

    def __init__(
        self,
        host: str = "10.42.0.1",
        port: int = 5555,
        reconnect_interval: float = 1.0,
    ) -> None:
        self._host = host
        self._port = port
        self._reconnect_interval = reconnect_interval

        self._lock = Lock()
        self._joint_positions: np.ndarray | None = None
        self._num_joints: int | None = None
        self._stop = Event()

        self._thread = Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info("NetworkDynamixelReader connecting to %s:%d", host, port)

    @property
    def num_joints(self) -> int:
        """Block until the server header is received, then return joint count."""
        while self._num_joints is None:
            if self._stop.is_set():
                raise RuntimeError("Reader closed before connection established")
            time.sleep(0.01)
        return self._num_joints

    def get_joint_positions(self) -> np.ndarray:
        """Return the latest joint positions in radians (blocks until first read)."""
        while self._joint_positions is None:
            if self._stop.is_set():
                raise RuntimeError("Reader closed before any data received")
            time.sleep(0.005)
        with self._lock:
            return self._joint_positions.copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        logger.info("NetworkDynamixelReader closed")

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._connect_and_stream()
            except Exception:
                if not self._stop.is_set():
                    logger.warning(
                        "Connection lost to %s:%d, reconnecting in %.1fs",
                        self._host,
                        self._port,
                        self._reconnect_interval,
                    )
                    self._stop.wait(self._reconnect_interval)

    def _connect_and_stream(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            sock.connect((self._host, self._port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            header = self._recvall(sock, 4)
            if header is None:
                return
            (n_joints,) = struct.unpack("!I", header)
            self._num_joints = n_joints
            frame_size = n_joints * 8
            logger.info(
                "Connected to %s:%d (%d joints)", self._host, self._port, n_joints
            )

            sock.settimeout(2.0)
            while not self._stop.is_set():
                data = self._recvall(sock, frame_size)
                if data is None:
                    break
                positions = np.frombuffer(data, dtype=">f8").copy()
                with self._lock:
                    self._joint_positions = positions
        finally:
            sock.close()

    def _recvall(self, sock: socket.socket, n: int) -> bytes | None:
        """Read exactly *n* bytes or return ``None`` on disconnect/timeout."""
        buf = bytearray()
        while len(buf) < n:
            if self._stop.is_set():
                return None
            try:
                chunk = sock.recv(n - len(buf))
            except socket.timeout:
                continue
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)
