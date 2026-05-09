import time
from typing import Any, Dict, Optional, Union

import dm_env
from loguru import logger

from limb.core.observation import (
    Observation,
    arm_obs_from_dict,
    camera_obs_from_dict,
)
from limb.robots.robot import Robot
from limb.robots.utils import Rate
from limb.sensors.cameras.camera import CameraDriver
from limb.utils.portal_utils import return_futures


class RobotEnv(dm_env.Environment):
    # Abstract methods.
    """A environment with a dm_env.Environment interface for a robot arm setup."""

    def __init__(
        self,
        robot_dict: Dict[str, Robot],
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        control_rate_hz: Union[Rate, float] = 100.0,
        use_joint_state_as_action: bool = False,
    ) -> None:
        self._robot_dict = robot_dict
        if isinstance(control_rate_hz, Rate):
            self._rate = control_rate_hz
        else:
            self._rate = Rate(control_rate_hz)

        self._use_joint_state_as_action = use_joint_state_as_action
        # get camera dict
        self._camera_dict = camera_dict
        # Per-stage timings from the most recent step(), in seconds.
        # Populated by step() / get_obs() and read by the control loop when
        # diagnosing slow iterations. Keys: "apply_action", "rate_sleep",
        # "obs_dispatch", "robot.<name>", "camera.<name>".
        self.last_step_timing: Dict[str, float] = {}

    def robot(self, name: str) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot_dict[name]

    def get_all_robots(self) -> Dict[str, Robot]:
        return self._robot_dict

    def __len__(self) -> int:
        return 0

    def _apply_action(self, action_dict: Dict[str, Any]) -> None:
        # Mode switches happen *before* position commands so a robot just
        # entering position_mode has its PD gains restored before its first
        # tracked target arrives.  `mode` is idempotent on the robot side and
        # is typically only set by composite agents on phase transitions.
        for name, action in action_dict.items():
            if isinstance(action, dict) and "mode" in action:
                mode = action["mode"]
                robot = self._robot_dict.get(name)
                if robot is None:
                    continue
                if mode == "zero_torque":
                    robot.zero_torque_mode()
                elif mode == "position":
                    robot.position_mode()
                elif mode == "compliant":
                    kd_scale = float(action.get("kd_scale", 0.1))
                    robot.damped_compliant_mode(kd_scale)
                else:
                    logger.warning(f"Unknown mode '{mode}' for robot '{name}' — ignoring")

        with return_futures(*self._robot_dict.values()):  # type: ignore
            for name, action in action_dict.items():
                if name == "base":
                    self._robot_dict[name].command_target_vel(action)
                elif self._use_joint_state_as_action:
                    self._robot_dict[name].command_joint_state(action)
                else:
                    if isinstance(action, dict) and "pos" not in action:
                        # Mode-only update (e.g. on a phase transition with no new target).
                        continue
                    self._robot_dict[name].command_joint_pos(action["pos"])

    def step(self, action: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Observation:  # type: ignore
        """Step the environment forward.

        Args:
            action: action to step the environment with.

        Returns:
            obs: typed Observation from the environment.
        """
        self.last_step_timing.clear()
        if len(action) != 0:
            t0 = time.perf_counter()
            self._apply_action(action)
            self.last_step_timing["apply_action"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._rate.sleep()
        self.last_step_timing["rate_sleep"] = time.perf_counter() - t0
        return self.get_obs()

    def get_obs(self) -> Observation:
        """Get observation from the environment.

        Returns:
            obs: typed Observation from the environment.
        """
        timestamp = time.time()

        assert self._camera_dict is not None, "Camera dictionary is not set."
        clients = list(self._camera_dict.values()) + list(self._robot_dict.values())

        camera_futures = {}
        robot_futures = {}
        t_dispatch = time.perf_counter()
        with return_futures(*clients):  # type: ignore
            for name, client in self._camera_dict.items():
                camera_data = client.read()
                camera_futures[name] = camera_data
            for name, robot in self._robot_dict.items():
                robot_obs = robot.get_observations()
                robot_futures[name] = robot_obs
        self.last_step_timing["obs_dispatch"] = time.perf_counter() - t_dispatch

        # Per-future result() timings are sequential, so they don't reflect
        # true wall-clock latency — but a stalled resource will still show up
        # as the one that consumed the bulk of the time.
        arms: Dict[str, Any] = {}
        for name, robot_obs_future in robot_futures.items():
            t0 = time.perf_counter()
            robot_obs = robot_obs_future.result()
            self.last_step_timing[f"robot.{name}"] = time.perf_counter() - t0
            arms[name] = arm_obs_from_dict(robot_obs)

        cameras: Dict[str, Any] = {}
        for name, camera_data_future in camera_futures.items():
            t0 = time.perf_counter()
            camera_data = camera_data_future.result()
            self.last_step_timing[f"camera.{name}"] = time.perf_counter() - t0
            cameras[name] = camera_obs_from_dict(camera_data)

        return Observation(
            timestamp=timestamp,
            arms=arms,
            cameras=cameras,
            extra={"timestamp_end": time.time()},
        )

    def reset(self) -> Observation:  # type: ignore
        return self.get_obs()

    def observation_spec(self):  # type: ignore
        return {}

    def action_spec(self):  # type: ignore
        spec = {}
        for name, robot in self._robot_dict.items():
            # if robot.get_robot_type() == RobotType.MOBILE_BASE:
            #     spec[name] = robot.joint_state_spec()
            # else:
            spec[name] = (
                robot.joint_state_spec() if self._use_joint_state_as_action else {"pos": robot.joint_pos_spec()}
            )
        return spec

    def close(self) -> None:
        assert self._camera_dict is not None, "Camera dictionary is not set."
        for camera_name, client in self._camera_dict.items():
            logger.debug(f"Closing camera {camera_name}")
            client.close()  # type: ignore

        logger.debug("Environment closed.")
