"""
Main launch script for YAM realtime robot control environment.
"""

import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tyro
from loguru import logger

from limb.agents.agent import Agent
from limb.core.observation import Observation, arm_obs_from_dict
from limb.envs.configs.instantiate import instantiate
from limb.envs.configs.loader import DictLoader
from limb.envs.robot_env import RobotEnv
from limb.recording.episode_recorder import EpisodeRecorder
from limb.recording.session import DataCollectionSession
from limb.robots.robot import Robot
from limb.robots.utils import Rate, Timeout
from limb.sensors.cameras.camera import CameraDriver
from limb.tui import StatusDisplay
from limb.utils.launch_utils import (
    cleanup_processes,
    initialize_agent,
    initialize_robots,
    initialize_sensors,
    run_server_proc,
    setup_can_interfaces,
    setup_logging,
)
from limb.visualization.viser_monitor import ViserMonitor

SAFE_MOVE_DURATION_S = 1.0
IK_WARMUP_TIMEOUT_S = 15.0
IK_WARMUP_POLL_S = 0.1

_shutdown_requested = False


def _sigint_handler(signum, frame):
    """Handle SIGINT by setting a flag instead of raising KeyboardInterrupt.

    This prevents the signal from propagating to Portal child processes
    and killing robot servers before we can do a safe shutdown.
    """
    global _shutdown_requested
    if _shutdown_requested:
        raise KeyboardInterrupt
    _shutdown_requested = True


@dataclass
class LaunchConfig:
    hz: float = 30.0
    cameras: Dict[str, Tuple[CameraDriver, int]] = field(default_factory=dict)
    robots: Dict[str, Union[str, Robot]] = field(default_factory=dict)
    max_steps: Optional[int] = None  # this is for testing
    save_path: Optional[str] = None
    station_metadata: Dict[str, str] = field(default_factory=dict)
    recording: Optional[Dict[str, Any]] = None  # EpisodeRecorder config (None = no recording)
    collection: Optional[Dict[str, Any]] = None  # DataCollectionSession config (managed episodes)
    sim_mode: bool = False  # skip CAN/sensors, instantiate robots & agent in-process
    enable_monitor: bool = True  # launch ViserMonitor for camera feeds + recording


@dataclass
class Args:
    config_path: Tuple[str, ...] = ("~/yam_realtime/configs/yam_viser_bimanual.yaml",)
    log_level: str = "INFO"


def _save_robot_positions(obs: Observation, robot_names: list) -> Dict[str, np.ndarray]:
    """Capture current joint positions (arm + gripper) from observations."""
    saved = {}
    for name in robot_names:
        arm = obs.arms.get(name)
        if arm is None:
            continue
        joint_pos = arm.joint_pos
        gripper_pos = arm.gripper_pos
        if joint_pos.size > 0:
            if gripper_pos is not None and gripper_pos.size > 0:
                saved[name] = np.concatenate([joint_pos, gripper_pos])
            else:
                saved[name] = joint_pos.copy()
    return saved


def _wait_for_ik_convergence(
    agent: Agent,
    obs: Observation,
    robot_names: list,
) -> Dict[str, Any]:
    """Poll agent.act() until the IK solver has fully converged.

    Convergence requires two conditions:
      1. All arm joints are non-zero (IK has started producing output).
      2. Consecutive readings are close (joints have stabilized).

    On first call the JAX JIT in pyroki can take several seconds to compile.
    """
    logger.info("Waiting for IK solver to warm up and converge...")
    obs_dict = obs.to_dict()
    deadline = time.time() + IK_WARMUP_TIMEOUT_S
    prev_joints: Dict[str, np.ndarray] = {}
    stable_count = 0
    STABLE_THRESHOLD = 5  # consecutive stable readings required

    while time.time() < deadline:
        action = agent.act(obs_dict)

        all_nonzero = True
        all_stable = True
        for name in robot_names:
            if name not in action or "pos" not in action[name]:
                all_nonzero = False
                break
            arm_joints = action[name]["pos"][:-1]
            if np.allclose(arm_joints, 0.0, atol=1e-4):
                all_nonzero = False
                break
            if name in prev_joints:
                if not np.allclose(arm_joints, prev_joints[name], atol=1e-3):
                    all_stable = False
            else:
                all_stable = False
            prev_joints[name] = arm_joints.copy()

        if all_nonzero and all_stable:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= STABLE_THRESHOLD:
            logger.info("IK solver converged (joints stabilized).")
            return action

        time.sleep(IK_WARMUP_POLL_S)

    logger.warning(f"IK solver did not fully converge within {IK_WARMUP_TIMEOUT_S}s, proceeding with current values.")
    return agent.act(obs_dict)


def _safe_move_robots(
    robots: Dict[str, Robot],
    targets: Dict[str, np.ndarray],
    duration_s: float = SAFE_MOVE_DURATION_S,
) -> None:
    """Slowly move robots to target joint positions using linear interpolation.

    All arms move simultaneously via threads (move_joints is a blocking RPC).
    """

    def _move_one(name: str, robot: Robot, target: np.ndarray) -> None:
        try:
            logger.info(f"Slowly moving '{name}' to target over {duration_s:.1f}s...")
            robot.move_joints(target, duration_s)
        except Exception as e:
            logger.warning(f"Could not slowly move '{name}': {e}")

    threads = []
    for name, robot in robots.items():
        if name not in targets:
            continue
        t = threading.Thread(target=_move_one, args=(name, robot, np.array(targets[name])), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=duration_s + 2.0)


SOFT_RELEASE_DURATION_S = 2.0


def _safe_release_robots(
    robots: Dict[str, Robot],
    duration_s: float = SOFT_RELEASE_DURATION_S,
) -> None:
    """Gradually fade gravity compensation then cut power on all robots."""

    def _release_one(name: str, robot: Robot) -> None:
        try:
            robot.soft_release(duration_s)
            logger.info(f"Soft-released '{name}' over {duration_s:.1f}s")
        except Exception as e:
            logger.warning(f"soft_release failed for '{name}', falling back to zero_torque_mode: {e}")
            try:
                robot.zero_torque_mode()
            except Exception:
                pass

    threads = []
    for name, robot in robots.items():
        t = threading.Thread(target=_release_one, args=(name, robot), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=duration_s + 2.0)


def main(args: Args) -> None:
    """
    Main launch entrypoint.

    1. Load configuration from yaml file
    2. Initialize sensors (cameras, force sensors, etc.)
    3. Setup CAN interfaces (for YAM communication)
    4. Initialize robots (hardware interface)
    5. Initialize agent (e.g. teleoperated control, policy control, etc.)
    6. Create environment
    7. Wait for IK solver to converge
    8. Slowly move to initial pose
    9. Run control loop (exits on SIGINT flag)
    10. On exit, slowly return to pre-teleop pose and release motors
    """
    global _shutdown_requested

    setup_logging(level=args.log_level)
    logger.info("Starting realtime control system...")

    server_processes = []
    saved_positions: Dict[str, np.ndarray] = {}
    robots: Dict[str, Robot] = {}

    # Install SIGINT handler BEFORE creating child processes so that
    # Ctrl+C sets a flag instead of killing Portal robot servers.
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        logger.info("Loading configuration...")
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])

        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        api_servers = configs_dict.pop("api_servers", None)

        server_procs = []

        if api_servers is not None:
            for api_server in api_servers:
                server_proc = run_server_proc(api_server)
                logger.info(f"API server {api_server} started")
                server_procs.append(server_proc)
        main_config = instantiate(configs_dict)

        # ----- Sim mode: everything runs in-process, no CAN/portal ----- #
        if main_config.sim_mode:
            logger.info("Running in sim mode (no CAN, no portal RPC)...")

            # Robots are already instantiated by instantiate() since they
            # were _target_ dicts in the YAML.
            robots = main_config.robots
            agent = instantiate(agent_cfg)

            display = StatusDisplay()
            display.start()
            logger.info("Starting sim control loop at %.1f Hz...", main_config.hz)
            try:
                _run_sim_control_loop(robots, agent, main_config, display=display)
            finally:
                display.stop()
            return

        # ----- Real hardware mode (original path) ----- #
        logger.info("Initializing sensors...")
        camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)

        setup_can_interfaces()

        logger.info("Initializing robots...")
        robots = initialize_robots(main_config.robots, server_processes)

        agent = initialize_agent(agent_cfg, server_processes)

        # Create a standalone ViserMonitor for agents that don't have their own
        # (e.g. GELLO, VR).  YamViserAgent already embeds a ViserMonitor.
        # Also create one for YamViserAgent when data collection is active,
        # since the agent's viser server runs in a Portal subprocess and we
        # can't attach GUI elements to it from the main process.
        monitor: Optional[ViserMonitor] = None
        agent_target = agent_cfg.get("_target_", "")
        has_collection = main_config.collection is not None
        needs_standalone_monitor = "YamViserAgent" not in agent_target
        needs_session_monitor = "YamViserAgent" in agent_target and has_collection

        if main_config.enable_monitor and (needs_standalone_monitor or needs_session_monitor):
            is_bimanual = len(robots) > 1
            right_extrinsic = (
                main_config.station_metadata.get("extrinsics", {}).get("right_arm_extrinsic")
                if main_config.station_metadata
                else None
            )
            monitor = ViserMonitor(
                enable_urdf=not needs_session_monitor,  # skip URDF if agent has its own
                bimanual=is_bimanual,
                right_arm_extrinsic=right_extrinsic,
            )
            label = "session panel + camera feeds" if needs_session_monitor else "camera feeds + recording + URDF"
            logger.info("ViserMonitor started (standalone) for {}", label)

        logger.info("Creating robot environment...")
        frequency = main_config.hz
        rate = Rate(frequency, rate_name="control_loop")

        env = RobotEnv(
            robot_dict=robots,
            camera_dict=camera_dict,
            control_rate_hz=rate,
        )

        # --- Safe startup ---
        obs = env.reset()
        saved_positions = _save_robot_positions(obs, list(robots.keys()))
        logger.info(f"Saved pre-teleop positions for: {list(saved_positions.keys())}")
        logger.info(f"Action spec: {env.action_spec()}")

        # Only IK-based agents (Viser, VR) need convergence warm-up.
        # Direct-joint agents (GELLO) produce valid actions immediately.
        _IK_AGENT_PATTERNS = ["YamViserAgent", "YamVrAgent"]
        agent_needs_ik = any(pat in agent_target for pat in _IK_AGENT_PATTERNS)

        if agent_needs_ik:
            initial_action = _wait_for_ik_convergence(agent, obs, list(robots.keys()))
        else:
            logger.info("Agent does not use IK, skipping convergence wait.")
            initial_action = agent.act(obs.to_dict())

        initial_targets = {}
        for name in robots:
            if name in initial_action and "pos" in initial_action[name]:
                initial_targets[name] = initial_action[name]["pos"]

        if initial_targets:
            logger.info("Moving to initial teleop pose (safe slow motion)...")
            _safe_move_robots(robots, initial_targets)

        # --- Episode recorder / collection session ---
        recorder: Optional[EpisodeRecorder] = None
        session: Optional[DataCollectionSession] = None
        if main_config.collection is not None:
            session = instantiate(main_config.collection)
            logger.info("DataCollectionSession configured (target={} episodes)", session.num_episodes)
        elif main_config.recording is not None:
            recorder = instantiate(main_config.recording)
            logger.info("EpisodeRecorder configured (base_dir={})", recorder.base_dir)

        # Use viser GUI panel when a monitor is available and session is active;
        # fall back to Rich terminal TUI otherwise.
        display: Any = None
        if monitor is not None and session is not None:
            from limb.recording.trigger import CompositeTrigger
            from limb.visualization.viser_session_panel import ViserSessionPanel

            panel = ViserSessionPanel(viser_server=monitor.viser_server)
            panel.start()
            display = panel
            # Hide ViserMonitor's standalone record button — session panel handles it
            try:
                if hasattr(monitor, "_record_button"):
                    monitor._record_button.visible = False
            except AttributeError:
                pass  # viser version may not support .visible
            # Compose GUI buttons with existing trigger (first signal wins)
            session.trigger = CompositeTrigger(sources=[panel, session.trigger])
            session.display = panel
        else:
            display = StatusDisplay()
            display.start()
            if session is not None:
                session.display = display

        logger.info("Starting control loop...")
        try:
            _run_control_loop(
                env, agent, main_config, monitor=monitor, recorder=recorder, session=session, display=display
            )
        finally:
            display.stop()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, initiating safe shutdown...")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise e
    finally:
        logger.info("Shutting down...")

        # Safe shutdown: return to pre-teleop positions and release motors.
        # Robot server processes are still alive because our SIGINT handler
        # prevented the signal from killing them.
        if saved_positions and robots:
            try:
                logger.info("Returning to pre-teleop positions (safe slow motion)...")
                _safe_move_robots(robots, saved_positions)
                _safe_release_robots(robots)
            except KeyboardInterrupt:
                logger.warning("Shutdown interrupted, cutting power immediately...")
                for name, robot in robots.items():
                    try:
                        robot.zero_torque_mode()
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Error during safe shutdown: {e}")

        if "session" in locals() and session is not None:
            session.close()
        elif "recorder" in locals() and recorder is not None:
            recorder.close()
        if "monitor" in locals() and monitor is not None:
            monitor.close()
        if "env" in locals():
            env.close()
        if "agent" in locals():
            cleanup_processes(agent, server_processes)

        signal.signal(signal.SIGINT, original_sigint)


def _run_sim_control_loop(
    robots: Dict[str, Robot],
    agent: Agent,
    config: LaunchConfig,
    display: Optional[StatusDisplay] = None,
) -> None:
    """Simplified control loop for sim mode (no portal, no cameras).

    Runs entirely in-process so the MuJoCo viewer stays on the main thread.
    """
    rate = Rate(config.hz, rate_name="sim_control_loop")
    steps = 0
    start_time = time.time()
    loop_count = 0

    def _build_sim_obs() -> Observation:
        arms = {name: arm_obs_from_dict(robot.get_observations()) for name, robot in robots.items()}
        return Observation(timestamp=time.time(), arms=arms)

    # Build initial observation from robots
    obs = _build_sim_obs()

    try:
        while True:
            # Check if any sim viewer has been closed
            for robot in robots.values():
                if hasattr(robot, "is_viewer_running") and not robot.is_viewer_running():
                    logger.info("Viewer closed, stopping...")
                    return

            action = agent.act(obs.to_dict())

            # Apply actions directly
            for name, act in action.items():
                if name in robots:
                    robots[name].command_joint_pos(act["pos"])

            rate.sleep()

            # Collect observations
            obs = _build_sim_obs()

            steps += 1
            loop_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                hz = loop_count / elapsed_time
                if display is not None:
                    display.update_loop(hz, steps)
                start_time = time.time()
                loop_count = 0

            if config.max_steps is not None and steps >= config.max_steps:
                logger.info(f"Reached max steps ({config.max_steps}), stopping...")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        if hasattr(agent, "close"):
            agent.close()
        for robot in robots.values():
            if hasattr(robot, "close"):
                robot.close()


def _run_control_loop(
    env: RobotEnv,
    agent: Agent,
    config: LaunchConfig,
    monitor: Optional[ViserMonitor] = None,
    recorder: Optional[EpisodeRecorder] = None,
    session: Optional[DataCollectionSession] = None,
    display: Optional[StatusDisplay] = None,
) -> None:
    """Run the main control loop.  Exits when _shutdown_requested is set by SIGINT."""
    steps = 0
    start_time = time.time()
    loop_count = 0

    obs = env.reset()

    while not _shutdown_requested:
        with Timeout(30, "Agent action"):
            action = agent.act(obs.to_dict())

        # Data collection session manages recording + trigger signals
        if session is not None:
            if not session.step(obs, action):
                break  # session complete or quit signal
        elif recorder is not None and recorder.is_recording:
            # Standalone recorder: record pre-step (s_t, a_t)
            recorder.record(obs, action)

        with Timeout(1, "Env step", "warning"):
            obs = env.step(action)

        if monitor is not None:
            monitor.update(obs)

        steps += 1
        loop_count += 1

        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            hz = loop_count / elapsed_time
            if display is not None:
                display.update_loop(hz, steps)
            start_time = time.time()
            loop_count = 0

        if config.max_steps is not None and steps >= config.max_steps:
            logger.info(f"Reached max steps ({config.max_steps}), stopping...")
            break

    if _shutdown_requested:
        logger.info("Shutdown flag detected, exiting control loop.")


if __name__ == "__main__":
    main(tyro.cli(Args))
