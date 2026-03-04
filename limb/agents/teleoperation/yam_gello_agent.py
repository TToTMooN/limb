"""
GELLO teleoperation agent for YAM arms using a Dynamixel-based leader device.

Uses direct joint-to-joint mapping from the Galaxea R1 Lite Teleop (or any
isomorphic GELLO device with Dynamixel servos) to the YAM follower arms.
No inverse kinematics is needed -- the leader and follower share the same
kinematic structure so joint angles transfer directly.

The server (``gello_position_server.py``) calibrates zero-pose offsets at
startup, so the positions received here are already calibrated joint angles
(0 = rest pose).  The agent sends them directly to the YAM, clamped to joint
limits.

Supports two connection modes:

* **USB serial** (direct):  Leader device plugged into the host PC.
* **Network** (Ethernet):  Leader device read by its onboard computer, which
  runs ``gello_position_server.py`` and streams positions over TCP.

Both single-arm and bimanual configurations are supported.

Usage
-----
Launch via YAML config::

    # Direct USB
    uv run limb/envs/launch.py --config_path configs/yam_gello_bimanual.yaml

    # Over network (R1 Lite Teleop Ethernet)
    uv run limb/envs/launch.py --config_path configs/yam_gello_network_bimanual.yaml
"""

import logging
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from dm_env.specs import Array

from limb.agents.agent import Agent
from limb.devices.dynamixel_reader import DynamixelReader
from limb.devices.network_dynamixel_reader import NetworkDynamixelReader
from limb.utils.portal_utils import remote

logger = logging.getLogger(__name__)

# YAM joint limits (radians) from robot_configs/yam/left.yaml
_YAM_JOINT_LIMITS = np.array(
    [
        [-2.09, 3.14],
        [0.00, 3.14],
        [0.05, 3.14],
        [-1.35, 1.35],
        [-1.50, 1.50],
        [-2.00, 2.00],
    ],
    dtype=np.float64,
)


class YamGelloAgent(Agent):
    """Teleoperate YAM arms via a Dynamixel GELLO leader device.

    The leader positions are assumed to be **calibrated** (zero = rest pose),
    either by ``gello_position_server.py`` (network mode) or by a
    ``DynamixelReader`` with pre-subtracted offsets (USB mode).

    Each ``act()`` call reads the leader joint angles and sends them directly
    to the YAM follower, clamped to joint limits::

        yam_target = clamp(leader_position, joint_limits)

    Parameters
    ----------
    port : str
        USB serial port for direct connection (e.g. ``"/dev/ttyUSB0"``).
        Ignored when *host* is set.
    baudrate : int
        Serial baudrate.  Default 4 000 000.  Ignored when *host* is set.
    host : str | None
        IP address of the R1 Lite Teleop running ``gello_position_server.py``.
        When set, positions are read over TCP instead of USB serial.
    network_port : int
        TCP port on the remote server (default 5555).
    bimanual : bool
        Whether to run in bimanual mode (two arms).
    left_motor_ids : Sequence[int]
        Dynamixel motor IDs for the left arm (base to tip).
    right_motor_ids : Sequence[int]
        Dynamixel motor IDs for the right arm (base to tip).
    joint_signs_left : Sequence[int]
        Per-joint sign correction for the left arm.  In USB mode these are
        passed to the ``DynamixelReader``; in network mode they are applied
        in ``act()`` to correct leader→follower direction differences.
    joint_signs_right : Sequence[int]
        Per-joint sign correction for the right arm.  Same behaviour as above.
    joycon_gripper : bool
        If True, spawn a ``JoyConGripperReader`` to read gripper values from
        paired Nintendo Joy-Con controllers via Bluetooth.  Requires the
        ``evdev`` package.
    default_gripper_value : float
        Gripper position sent every step when *joycon_gripper* is False
        (0.0 = open).
    """

    use_joint_state_as_action: bool = False

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 4_000_000,
        host: Optional[str] = None,
        network_port: int = 5555,
        bimanual: bool = False,
        left_motor_ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
        right_motor_ids: Sequence[int] = (7, 8, 9, 10, 11, 12),
        joint_signs_left: Sequence[int] = (1, 1, -1, -1, -1, 1),
        joint_signs_right: Sequence[int] = (1, 1, -1, -1, -1, 1),
        joycon_gripper: bool = False,
        default_gripper_value: float = 0.0,
    ) -> None:
        self.bimanual = bimanual
        self._default_gripper = default_gripper_value
        self._joint_limits = _YAM_JOINT_LIMITS
        self._signs_left = np.asarray(joint_signs_left, dtype=np.float64)
        self._signs_right = np.asarray(joint_signs_right, dtype=np.float64)

        self._gripper_reader: Optional["JoyConGripperReader"] = None
        if joycon_gripper:
            from limb.devices.joycon_gripper_reader import JoyConGripperReader

            self._gripper_reader = JoyConGripperReader()
            logger.info("Joy-Con gripper control enabled")

        self._n_left = len(left_motor_ids)
        self._n_right = len(right_motor_ids) if bimanual else 0

        self._reader: Union[DynamixelReader, NetworkDynamixelReader]
        self._network_mode = host is not None
        if self._network_mode:
            self._reader = NetworkDynamixelReader(host=host, port=network_port)
            logger.info(
                "YamGelloAgent using network reader at %s:%d (bimanual=%s)",
                host,
                network_port,
                bimanual,
            )
        else:
            all_ids = list(left_motor_ids) + (list(right_motor_ids) if bimanual else [])
            all_signs = list(joint_signs_left) + (list(joint_signs_right) if bimanual else [])
            self._reader = DynamixelReader(
                port=port,
                motor_ids=all_ids,
                joint_signs=all_signs,
                baudrate=baudrate,
            )
            logger.info(
                "YamGelloAgent using USB reader (bimanual=%s, port=%s, left_ids=%s%s)",
                bimanual,
                port,
                list(left_motor_ids),
                f", right_ids={list(right_motor_ids)}" if bimanual else "",
            )

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        leader_pos = self._reader.get_joint_positions()

        if self._gripper_reader is not None:
            grip_left, grip_right = self._gripper_reader.get_gripper_values()
        else:
            grip_left = grip_right = self._default_gripper

        left_joints = leader_pos[: self._n_left]
        if self._network_mode:
            left_joints = left_joints * self._signs_left
        left_clamped = np.clip(left_joints, self._joint_limits[:, 0], self._joint_limits[:, 1])

        action: Dict[str, Dict[str, np.ndarray]] = {
            "left": {
                "pos": np.concatenate([left_clamped, [grip_left]]),
            }
        }

        if self.bimanual:
            right_joints = leader_pos[self._n_left : self._n_left + self._n_right]
            if self._network_mode:
                right_joints = right_joints * self._signs_right
            right_clamped = np.clip(right_joints, self._joint_limits[:, 0], self._joint_limits[:, 1])
            action["right"] = {
                "pos": np.concatenate([right_clamped, [grip_right]]),
            }

        return action

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        spec: Dict[str, Dict[str, Array]] = {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
        }
        if self.bimanual:
            spec["right"] = {"pos": Array(shape=(7,), dtype=np.float32)}
        return spec

    def close(self) -> None:
        self._reader.close()
        if self._gripper_reader is not None:
            self._gripper_reader.close()
        logger.info("YamGelloAgent closed")
