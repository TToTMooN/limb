#!/usr/bin/env python3
"""
TCP server that streams calibrated Dynamixel joint positions from the R1 Lite Teleop.

v2 — wedge-proof rewrite of the serving loop (protocol unchanged):

  * NEWEST CLIENT WINS: a new connection immediately replaces the current
    one.  A stale/leaked client (e.g. the workstation DLP's transparent
    proxy leaking its upstream leg and never reading — observed buffering
    8.8 MB unread) can no longer hold the server captive: the reader's
    reconnect is accepted instantly and the old connection is closed.
  * SEND TIMEOUT (2 s) on the client socket: a peer whose receive window
    fills can only stall one send cycle, never block ``sendall`` forever.
  * TCP keepalive + user timeout so dead peers are reaped at kernel level.
  * ``accept()`` hardened against ECONNABORTED from connections reset while
    queued (previously uncaught -> server process death).

Runs on the R1 Lite Teleop onboard computer (cat@10.42.0.1).  Reads joint
positions from the Dynamixel servo chain via USB serial, subtracts zero-pose
offsets captured at startup, and streams calibrated joint angles to the most
recently connected client over TCP.

Startup procedure:

    1. Place both teleop arms in the ZERO / REST POSE (all linkages in the
       positioning slots, as shown in the Galaxea documentation).
    2. Start the server — it reads the current positions and stores them as
       zero-pose offsets.
    3. After "Calibrated. Offsets: ..." appears, the arms are free to move.

Deployment (replaces ~/gello_position_server.py on the device)::

    scp scripts/gello_position_server_v2.py cat@10.42.0.1:~/gello_position_server.py
    ssh cat@10.42.0.1 "python3 ~/gello_position_server.py"

Protocol (unchanged from v1)::

    1. Client connects via TCP.
    2. Server sends 4-byte header: uint32 big-endian = number of joints (N).
    3. Server then continuously sends frames of N * 8 bytes (float64 big-endian)
       at the configured rate (default 200 Hz).  Values are calibrated joint
       angles in radians (0 = zero/rest pose).
    4. A newer client connection supersedes the current one.

Compatibility:
    - Does NOT interfere with node_monitor.service or any ROS 2 nodes.
    - Only accesses /dev/r1litet_usb (Dynamixel USB serial).
    - Cannot run simultaneously with the Galaxea teleop session
      (robot_startup.sh) since both use /dev/r1litet_usb.

Requirements (all pre-installed on the device):
    - Python 3.10+
    - numpy
    - dynamixel_sdk (via /opt/ros/humble)
"""

import argparse
import logging
import select
import socket
import struct
import sys
import time
from threading import Event, Lock, Thread

import numpy as np

sys.path.insert(0, "/opt/ros/humble/local/lib/python3.10/dist-packages")

from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncRead,
    PacketHandler,
    PortHandler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gello_position_server")

ADDR_PRESENT_POSITION = 140
LEN_PRESENT_POSITION = 4
ADDR_TORQUE_ENABLE = 64

# LubanCat-4 board (current R1 Lite Teleop hardware)
LUBANCAT4_JOINT_SIGNS = [1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]
DEFAULT_JOINT_SIGNS = [1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1]

SEND_TIMEOUT_S = 2.0  # max time one frame send may block before the client is dropped


def detect_lubancat4():
    try:
        with open("/proc/device-tree/model", "rb") as f:
            model = f.read().decode("utf-8", errors="ignore").replace("\0", "").strip()
        return "LubanCat-4" in model, model
    except Exception:
        return False, "unknown"


class DynamixelPoller:
    """Reads Dynamixel joint positions with zero-pose calibration."""

    def __init__(self, device, motor_ids, joint_signs, baudrate):
        self._ids = list(motor_ids)
        self._signs = np.asarray(joint_signs, dtype=np.float64)
        self._n = len(motor_ids)

        self._port = PortHandler(device)
        self._pkt = PacketHandler(2.0)

        if not self._port.openPort():
            raise RuntimeError(f"Cannot open {device}")
        if not self._port.setBaudRate(baudrate):
            self._port.closePort()
            raise RuntimeError(f"Cannot set baudrate {baudrate}")

        for did in self._ids:
            self._pkt.write1ByteTxRx(self._port, did, ADDR_TORQUE_ENABLE, 0)

        self._sync = GroupSyncRead(
            self._port, self._pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        for did in self._ids:
            if not self._sync.addParam(did):
                self._port.closePort()
                raise RuntimeError(f"Cannot add motor {did} to sync read")

        self._offsets = np.zeros(self._n, dtype=np.float64)
        self._lock = Lock()
        self._positions = None
        self._stop = Event()

    @property
    def num_joints(self):
        return self._n

    def _read_raw_once(self):
        """Single blocking read of signed angles (no offset subtraction)."""
        use_fast = hasattr(self._sync, "fastSyncRead")
        for _ in range(50):
            rc = self._sync.fastSyncRead() if use_fast else self._sync.txRxPacket()
            if rc == COMM_SUCCESS:
                raw = np.empty(self._n, dtype=np.int32)
                for i, did in enumerate(self._ids):
                    val = self._sync.getData(
                        did, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                    )
                    raw[i] = np.int32(np.uint32(val))
                return raw / 2048.0 * np.pi * self._signs
            time.sleep(0.01)
        return None

    def calibrate(self):
        """Capture zero-pose offsets.  Call with arms in the rest position."""
        logger.info("Capturing zero-pose offsets (arms must be in rest position)...")
        pos = self._read_raw_once()
        if pos is None:
            raise RuntimeError("Failed to read positions for calibration")
        self._offsets = pos.copy()
        logger.info("Calibrated. Offsets: %s", np.round(self._offsets, 3).tolist())

    def start(self):
        """Start the background polling thread (call after calibrate)."""
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Polling %d motors", self._n)

    def get(self):
        with self._lock:
            return self._positions

    def close(self):
        self._stop.set()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)
        self._port.closePort()

    def _loop(self):
        use_fast = hasattr(self._sync, "fastSyncRead")
        while not self._stop.is_set():
            time.sleep(0.001)
            try:
                rc = (
                    self._sync.fastSyncRead()
                    if use_fast
                    else self._sync.txRxPacket()
                )
                if rc != COMM_SUCCESS:
                    continue
                raw = np.empty(self._n, dtype=np.int32)
                for i, did in enumerate(self._ids):
                    val = self._sync.getData(
                        did, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                    )
                    raw[i] = np.int32(np.uint32(val))
                angles = raw / 2048.0 * np.pi * self._signs - self._offsets
                with self._lock:
                    self._positions = angles
            except Exception:
                logger.exception("Read error")
                time.sleep(0.01)


def _configure_client(conn):
    """Timeouts + keepalive so a dead or non-reading peer can't wedge us."""
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(SEND_TIMEOUT_S)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 2)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 2)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
        # Drop the connection if transmitted data stays unacknowledged 5 s.
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 5000)
    except (AttributeError, OSError):
        pass  # older kernels/platforms: keepalive defaults still apply


def serve(poller, tcp_port, hz):
    """Stream calibrated positions to the most recently connected client."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", tcp_port))
    srv.listen(2)
    logger.info(
        "Listening on 0.0.0.0:%d (streaming at %d Hz, newest client wins)", tcp_port, hz
    )

    interval = 1.0 / hz
    header = struct.pack("!I", poller.num_joints)
    conn = None
    addr = None

    while True:
        # Poll for new connections even while serving — the newest client
        # always takes over, so a stale peer can never hold the server.
        wait = 0.0 if conn is not None else 0.5
        try:
            readable, _, _ = select.select([srv], [], [], wait)
        except OSError:
            readable = []
        if readable:
            try:
                new_conn, new_addr = srv.accept()
            except OSError as e:
                # e.g. ECONNABORTED: connection reset while queued — ignore.
                logger.warning("accept() failed: %s", e)
                new_conn = None
            if new_conn is not None:
                if conn is not None:
                    logger.info("Client %s superseded by %s", addr, new_addr)
                    try:
                        conn.close()
                    except OSError:
                        pass
                conn, addr = new_conn, new_addr
                _configure_client(conn)
                logger.info("Client connected: %s", addr)
                try:
                    conn.sendall(header)
                except (socket.timeout, OSError):
                    logger.info("Client vanished before header: %s", addr)
                    conn.close()
                    conn = None

        if conn is None:
            continue

        t0 = time.monotonic()
        pos = poller.get()
        try:
            if pos is not None:
                conn.sendall(pos.astype(">f8").tobytes())
        except (socket.timeout, BrokenPipeError, ConnectionResetError, OSError):
            logger.info("Client disconnected: %s", addr)
            try:
                conn.close()
            except OSError:
                pass
            conn = None
            continue
        elapsed = time.monotonic() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)


def main():
    is_lbc4, model = detect_lubancat4()
    default_signs = LUBANCAT4_JOINT_SIGNS if is_lbc4 else DEFAULT_JOINT_SIGNS

    p = argparse.ArgumentParser(
        description="Stream calibrated Dynamixel joint positions over TCP"
    )
    p.add_argument(
        "--device",
        default="/dev/r1litet_usb",
        help="Dynamixel USB serial device (default: /dev/r1litet_usb)",
    )
    p.add_argument(
        "--baudrate",
        type=int,
        default=4_000_000,
        help="Serial baudrate (default: 4000000)",
    )
    p.add_argument(
        "--tcp-port",
        type=int,
        default=5555,
        help="TCP port to listen on (default: 5555)",
    )
    p.add_argument(
        "--hz",
        type=int,
        default=200,
        help="Streaming rate in Hz (default: 200)",
    )
    p.add_argument(
        "--motor-ids",
        type=int,
        nargs="+",
        default=list(range(1, 13)),
        help="Dynamixel motor IDs (default: 1..12)",
    )
    p.add_argument(
        "--joint-signs",
        type=int,
        nargs="+",
        default=None,
        help="Per-joint sign (+1/-1). Default: auto-detect from board",
    )
    args = p.parse_args()

    signs = args.joint_signs or default_signs[: len(args.motor_ids)]
    if len(signs) != len(args.motor_ids):
        p.error("--joint-signs length must match --motor-ids length")

    logger.info("Board: %s (LubanCat-4: %s)", model, is_lbc4)
    logger.info("Joint signs: %s", signs)

    poller = DynamixelPoller(args.device, args.motor_ids, signs, args.baudrate)
    try:
        poller.calibrate()
        poller.start()
        serve(poller, args.tcp_port, args.hz)
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        poller.close()


if __name__ == "__main__":
    main()
