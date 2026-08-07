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
    first_data_timeout : float
        Max seconds :meth:`get_joint_positions` / :attr:`num_joints` wait for
        the *first* data after startup before raising.  A wedged or
        client-occupied server otherwise blocks the caller forever with no
        indication of what's wrong.
    stale_timeout : float
        Max seconds without a frame before the current connection is declared
        dead and re-established.  Network disruptions (e.g. the host DLP's
        periodic tc-filter churn on the robot link) can silently black-hole an
        established stream — without this, the reader would retry ``recv`` on
        the dead socket forever, serving frozen positions with no warning.
    """

    def __init__(
        self,
        host: str = "10.42.0.1",
        port: int = 5555,
        reconnect_interval: float = 1.0,
        first_data_timeout: float = 15.0,
        stale_timeout: float = 2.0,
    ) -> None:
        self._host = host
        self._port = port
        self._reconnect_interval = reconnect_interval
        self._first_data_timeout = first_data_timeout
        self._stale_timeout = stale_timeout
        self._started_at = time.monotonic()
        self._last_data_ts: float | None = None

        self._lock = Lock()
        self._joint_positions: np.ndarray | None = None
        self._num_joints: int | None = None
        self._stop = Event()

        self._thread = Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info("NetworkDynamixelReader connecting to %s:%d", host, port)

    def _check_first_data_deadline(self, what: str) -> None:
        if time.monotonic() - self._started_at > self._first_data_timeout:
            raise TimeoutError(
                f"No {what} from GELLO server at {self._host}:{self._port} within "
                f"{self._first_data_timeout:.0f}s. The server may be down, wedged, or still "
                f"streaming to a stale client. Restart it: bash scripts/start_gello_server.sh"
            )

    @property
    def num_joints(self) -> int:
        """Block until the server header is received, then return joint count."""
        while self._num_joints is None:
            if self._stop.is_set():
                raise RuntimeError("Reader closed before connection established")
            self._check_first_data_deadline("header")
            time.sleep(0.01)
        return self._num_joints

    def get_joint_positions(self) -> np.ndarray:
        """Return the latest joint positions in radians (blocks until first read).

        Raises ``TimeoutError`` if no data ever arrives within
        ``first_data_timeout`` seconds of reader startup.
        """
        while self._joint_positions is None:
            if self._stop.is_set():
                raise RuntimeError("Reader closed before any data received")
            self._check_first_data_deadline("data")
            time.sleep(0.005)
        with self._lock:
            return self._joint_positions.copy()

    def seconds_since_data(self) -> float:
        """Age of the most recent frame, in seconds (``inf`` before first data).

        Lets callers distinguish live positions from stale ones held over a
        connection outage (e.g. to freeze/smooth follower motion).
        """
        if self._last_data_ts is None:
            return float("inf")
        return time.monotonic() - self._last_data_ts

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

            # Bound the header wait: if the server accepted us but never sends
            # (wedged on a stale client), give up and reconnect rather than
            # spinning on this socket forever.
            sock.settimeout(0.5)
            header = self._recvall(sock, 4, max_wait=self._stale_timeout)
            if header is None:
                return
            (n_joints,) = struct.unpack("!I", header)
            self._num_joints = n_joints
            frame_size = n_joints * 8
            logger.info(
                "Connected to %s:%d (%d joints)", self._host, self._port, n_joints
            )

            while not self._stop.is_set():
                data = self._recvall(sock, frame_size, max_wait=self._stale_timeout)
                if data is None:
                    if not self._stop.is_set():
                        logger.warning(
                            "GELLO stream from %s:%d went silent for %.1fs — reconnecting",
                            self._host,
                            self._port,
                            self._stale_timeout,
                        )
                    break
                positions = np.frombuffer(data, dtype=">f8").copy()
                with self._lock:
                    self._joint_positions = positions
                    self._last_data_ts = time.monotonic()
        finally:
            sock.close()

    def _recvall(self, sock: socket.socket, n: int, max_wait: float | None = None) -> bytes | None:
        """Read exactly *n* bytes.

        Returns ``None`` on disconnect, stop, or when *max_wait* seconds pass
        without completing the read (stale/black-holed connection).
        """
        buf = bytearray()
        start = time.monotonic()
        while len(buf) < n:
            if self._stop.is_set():
                return None
            if max_wait is not None and time.monotonic() - start > max_wait:
                return None
            try:
                chunk = sock.recv(n - len(buf))
            except socket.timeout:
                continue
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)
