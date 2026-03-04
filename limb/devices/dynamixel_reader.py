"""
Minimal Dynamixel servo position reader for GELLO-style teleoperation devices.

Reads joint positions from Dynamixel XM/XL series servos over USB serial using
the Dynamixel SDK Protocol 2.0 GroupSyncRead.  A background thread polls at
~1 kHz so the main control loop can always grab the latest positions without
blocking.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Sequence

import numpy as np
from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncRead,
    PacketHandler,
    PortHandler,
)

logger = logging.getLogger(__name__)

ADDR_PRESENT_POSITION = 140
LEN_PRESENT_POSITION = 4
ADDR_TORQUE_ENABLE = 64


class DynamixelReader:
    """Continuously reads joint positions from a chain of Dynamixel servos.

    Parameters
    ----------
    port : str
        Serial port path (e.g. ``"/dev/ttyUSB0"``).
    motor_ids : Sequence[int]
        Dynamixel motor IDs to read, in order.
    joint_signs : Sequence[int]
        Per-joint sign multiplier (+1 or -1) to correct motor direction.
    baudrate : int
        Serial baudrate.  The R1 Lite Teleop uses 4 000 000.
    """

    def __init__(
        self,
        port: str,
        motor_ids: Sequence[int],
        joint_signs: Sequence[int],
        baudrate: int = 4_000_000,
    ) -> None:
        if len(motor_ids) != len(joint_signs):
            raise ValueError(
                f"motor_ids ({len(motor_ids)}) and joint_signs ({len(joint_signs)}) must have same length"
            )

        self._ids = list(motor_ids)
        self._signs = np.asarray(joint_signs, dtype=np.float64)
        self._num_joints = len(motor_ids)

        self._port_handler = PortHandler(port)
        self._packet_handler = PacketHandler(2.0)

        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open Dynamixel port: {port}")
        if not self._port_handler.setBaudRate(baudrate):
            self._port_handler.closePort()
            raise RuntimeError(f"Failed to set baudrate to {baudrate}")

        self._sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        for dxl_id in self._ids:
            if not self._sync_read.addParam(dxl_id):
                self._port_handler.closePort()
                raise RuntimeError(f"Failed to add Dynamixel ID {dxl_id} to sync read")

        self._disable_torque()

        self._lock = Lock()
        self._joint_positions: np.ndarray | None = None
        self._stop = Event()
        self._thread = Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info("DynamixelReader started on %s (IDs: %s, baudrate: %d)", port, motor_ids, baudrate)

    @property
    def num_joints(self) -> int:
        return self._num_joints

    def get_joint_positions(self) -> np.ndarray:
        """Return the latest signed joint positions in radians.

        Blocks until at least one successful read has completed.
        """
        while self._joint_positions is None:
            if self._stop.is_set():
                raise RuntimeError("Reader was closed before any successful read")
            time.sleep(0.005)
        with self._lock:
            return self._joint_positions.copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._port_handler.closePort()
        logger.info("DynamixelReader closed")

    def _disable_torque(self) -> None:
        for dxl_id in self._ids:
            self._packet_handler.write1ByteTxRx(self._port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)

    def _read_loop(self) -> None:
        use_fast = hasattr(self._sync_read, "fastSyncRead")
        if use_fast:
            logger.info("Using fastSyncRead (available in this dynamixel_sdk version)")
        else:
            logger.info("fastSyncRead not available, using txRxPacket")

        while not self._stop.is_set():
            time.sleep(0.001)
            try:
                result = self._sync_read.fastSyncRead() if use_fast else self._sync_read.txRxPacket()
                if result != COMM_SUCCESS:
                    continue

                raw = np.empty(self._num_joints, dtype=np.int32)
                for i, dxl_id in enumerate(self._ids):
                    val = self._sync_read.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    raw[i] = np.int32(np.uint32(val))

                angles = raw / 2048.0 * np.pi * self._signs

                with self._lock:
                    self._joint_positions = angles
            except Exception:
                logger.exception("DynamixelReader read error")
                time.sleep(0.01)
