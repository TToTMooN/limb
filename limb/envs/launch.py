"""
Main launch script for YAM realtime robot control environment.
"""

import os
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

ObsPreprocess = Callable[[Dict[str, Any]], Dict[str, Any]]

import numpy as np
import tyro
from loguru import logger

from limb import ROOT_PATH
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
SLOW_STEP_WARN_S = 1.0
SLOW_HZ_FRACTION = 0.8
SLOW_HZ_STREAK_S = 3
SLOW_HZ_REPEAT_S = 10

_shutdown_requested = False


def _build_obs_image_preprocess(agent_cfg: Dict[str, Any]) -> Optional[ObsPreprocess]:
    """Build a function that resizes obs camera images in the main process so
    Portal RPC ships a smaller payload to the agent process.

    Without this, native 480x640x3 frames from three cameras (~2.7 MB) cross
    the Portal channel every tick at the full control rate. Pre-resizing on
    the launch side cuts this by 10-12x and shaves measurable ms off the
    per-tick budget on local-host loopback.

    The resize is *exactly* what the agent's ``obs_transform`` would do
    anyway: limb's ``ObsTransform`` and ``OpenPIObsTransform`` both check
    ``img.shape == target`` and skip work if it matches, so this is idempotent
    (pre-resize + agent-side resize is the same as agent-side resize alone).

    Returns
    -------
    callable | None
        Returns ``preprocess(obs_dict) -> obs_dict``, or ``None`` when the
        agent doesn't declare an ``obs_transform.image_size`` (e.g. teleop
        agents that don't consume cameras).
    """
    import cv2

    # Find the obs_transform — direct on the agent, or nested inside a
    # composite agent's inner_policy (e.g. DAggerAgent wrapping YamPolicyAgent).
    def _find_obs_transform(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        xf = cfg.get("obs_transform")
        if isinstance(xf, dict):
            return xf
        # composite agents: DAggerAgent wraps `inner_policy`; SubtaskRLAgent wraps `vla`
        for key in ("inner_policy", "vla"):
            inner = cfg.get(key)
            if isinstance(inner, dict):
                found = _find_obs_transform(inner)
                if found is not None:
                    return found
        return None

    obs_xform_cfg = _find_obs_transform(agent_cfg)
    if obs_xform_cfg is None:
        return None
    image_size = obs_xform_cfg.get("image_size")
    if image_size is None or len(image_size) != 2:
        return None
    h, w = int(image_size[0]), int(image_size[1])

    xform_target = obs_xform_cfg.get("_target_", "")
    if "SubtaskRLAgent" in str(agent_cfg.get("_target_", "")):
        # SubRL: the POLICY gets the padded 224 every tick (exactly the old, smooth
        # RPC payload), and the coding agents' PERCEPTION gets an aspect-preserving
        # 448 frame attached under images["rgb_hd"] only every hd_every-th tick.
        # Rationale: 224 destroys the vial for SAM3 (22:47 run: fallen never
        # escalated), but shipping 448 EVERY tick was 1.35 MB/tick of Portal payload
        # — launch-side agent.act spiked to 170-440 ms and the loop fell to 9-17 Hz
        # (23:15/23:40 runs) even though the agent's internal stages were <2 ms.
        # The detector is throttled to ~0.5 Hz per camera, so ~2 Hz of HD frames
        # (cached agent-side between arrivals) loses nothing.
        from limb.agents.policy_learning.transforms import _resize_with_pad
        hd_every = 15
        state = {"n": 0}

        def _hd(img):
            H, W = img.shape[:2]
            m = max(H, W)
            if m <= 448:
                return img
            s = 448.0 / m
            return cv2.resize(img, (int(round(W * s)), int(round(H * s))),
                              interpolation=cv2.INTER_LINEAR)

        def preprocess(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
            state["n"] += 1
            attach_hd = (state["n"] % hd_every) == 1
            for value in obs_dict.values():
                if isinstance(value, dict):
                    # Belt-and-suspenders vs the depth_data ALIAS (review 2026-08-05):
                    # robocam duplicates depth into a top-level key that would ride
                    # the act() RPC every tick — RealsenseDepthLite already nulls it
                    # at the source; strip here so no camera class can regress this.
                    value.pop("depth_data", None)
                    imgs = value.get("images")
                    if isinstance(imgs, dict) and "rgb" in imgs:
                        raw = imgs["rgb"]
                        if attach_hd:
                            imgs["rgb_hd"] = _hd(raw)
                        elif "depth" in imgs:
                            # Depth mode (2026-08-05): aligned depth rides ONLY the
                            # HD ticks — same Portal-budget rule as rgb_hd, and it
                            # keeps the depth frame paired with the agent's cached
                            # rgb_hd from the same tick. (Camera-side downscale in
                            # RealsenseDepthLite already matched the hd geometry.)
                            del imgs["depth"]
                        imgs["rgb"] = _resize_with_pad(raw, h, w)
            return obs_dict

        return preprocess
    if "OpenPIObsTransform" in xform_target:
        # OpenPI uses aspect-preserving padded resize. Mirror it here exactly
        # so the agent-side resize is a no-op.
        from limb.agents.policy_learning.transforms import _resize_with_pad

        def _resize(img):
            return _resize_with_pad(img, h, w)
    else:
        # Default to plain bilinear resize (ObsTransform, also fine for any
        # untargeted/custom transform that just wants shape-matched input).
        def _resize(img):
            if img.shape[0] == h and img.shape[1] == w:
                return img
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    def preprocess(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Walk the top level; camera entries are dicts with an "images" sub-dict
        # whose "rgb" key holds the raw frame. (Same layout as Observation.to_dict.)
        for value in obs_dict.values():
            if isinstance(value, dict):
                imgs = value.get("images")
                if isinstance(imgs, dict) and "rgb" in imgs:
                    imgs["rgb"] = _resize(imgs["rgb"])
        return obs_dict

    return preprocess


def _resolve_robot_configs(robots_cfg: Any) -> Dict[str, Any]:
    """Resolve each robot's YAML path(s) into a merged config dict.

    Stored in episode metadata so `limb replay` can reconstruct the exact
    hardware the episode was recorded on, without needing a launch config.

    Accepts any of the forms supported by `_create_robot_client`:
      - single YAML path: ``"robot_configs/yam/left.yaml"``
      - list/tuple of YAML paths (merged in order): ``["a.yaml", "b.yaml"]``
      - inline dict with ``_target_`` (no disk load needed)
      - omegaconf ListConfig/DictConfig (coerced to native types)
    """
    import omegaconf

    resolved: Dict[str, Any] = {}
    for name, entry in robots_cfg.items():
        try:
            if isinstance(entry, omegaconf.dictconfig.DictConfig):
                entry = omegaconf.OmegaConf.to_container(entry, resolve=True)
            if isinstance(entry, omegaconf.listconfig.ListConfig):
                entry = list(entry)

            if isinstance(entry, dict):
                # Already a fully-resolved config dict (inline `_target_`)
                resolved[name] = dict(entry)
            elif isinstance(entry, str):
                resolved[name] = DictLoader.load(entry)
            elif isinstance(entry, (list, tuple)):
                resolved[name] = DictLoader.load(list(entry))
            else:
                logger.warning(
                    "Unrecognised robot config entry type for '{}': {!r}. "
                    "Episode metadata will omit this arm's robot_configs.",
                    name,
                    type(entry).__name__,
                )
        except Exception as e:
            logger.warning("Could not resolve robot config for '{}': {}", name, e)
    return resolved


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
    # Robots placed in zero_torque_mode immediately after init -- typically the
    # operator-backdriven leader arms in a bilateral / DAgger setup.  These
    # robots are also excluded from the "return to startup pose" shutdown step.
    release_at_startup: List[str] = field(default_factory=list)
    # When > 0, released robots enter damped_compliant_mode(release_kd_scale)
    # instead of zero_torque_mode -- gravity comp still active, but kd is
    # restored to a fraction of the YAML-configured value to suppress ringing
    # while the operator backdrives.  Typical range: 0.05 - 0.20.  Mirrors
    # DAggerAgent.correcting_kd_scale for the bilateral release path.
    release_kd_scale: float = 0.0


@dataclass
class Args:
    config_path: Tuple[str, ...] = ("~/yam_realtime/configs/yam_viser_bimanual.yaml",)
    log_level: str = "INFO"
    # Perception VLM override for every make_vial_detector block in the loaded config
    # (SubRL coding agents). e.g. --vlm gpt-5.5 | --vlm gemini-er | --vlm sam3_owlvit.
    # None = use whatever the config says.
    vlm: Optional[str] = None


def _normalize_vlm(name: str) -> Tuple[str, Optional[str]]:
    """User-friendly VLM name -> (detector backend, optional model override)."""
    n = name.strip().lower().replace("-", "_")
    if n.startswith("gpt") or n == "openai":
        model = name.strip().lower().replace("_", "-")
        return "gpt", (model if any(c.isdigit() for c in model) else None)
    if n.startswith("gemini"):
        return "gemini_er", None
    if "owl" in n:
        return "sam3_owlvit", None
    if "sam" in n:
        return "sam3", None            # SAM 3 text-prompted segmentation + geometric pose
    raise ValueError(f"--vlm {name!r}: expected gpt-5.5 / gemini-er / sam3 / sam3_owlvit")


def _override_vlm(node: Any, backend: str, model: Optional[str]) -> int:
    """Recursively set backend/model on every make_vial_detector block. Returns the
    number of blocks overridden."""
    from omegaconf import DictConfig, ListConfig
    count = 0
    if isinstance(node, (dict, DictConfig)):
        try:
            target = str(node.get("_target_", ""))
        except Exception:
            target = ""
        if target.endswith("make_vial_detector"):
            node["backend"] = backend
            node["model"] = model           # None clears a stale override from another backend
            node["request_timeout_s"] = max(float(node.get("request_timeout_s", 10.0) or 10.0), 30.0)
            count += 1
        for v in list(node.values()):
            count += _override_vlm(v, backend, model)
    elif isinstance(node, (list, tuple, ListConfig)):
        for v in node:
            count += _override_vlm(v, backend, model)
    return count


def _agent_query_bool(agent: Any, method_name: str) -> bool:
    """Call ``agent.<method_name>()`` if exposed; return False otherwise.

    Works for both in-process agents and Portal-RPC Client handles.  A Client
    raises AttributeError on unknown methods (no falling back to attr lookup),
    so we introspect ``supported_remote_methods`` first.  In-process agents
    don't have that attribute; we fall back to a normal hasattr check.
    """
    rpc_methods = getattr(agent, "supported_remote_methods", None)
    if rpc_methods is not None:
        if method_name in rpc_methods:
            try:
                return bool(getattr(agent, method_name)())
            except Exception:
                return False
        return False
    method = getattr(agent, method_name, None)
    if method is None:
        return False
    if callable(method):
        try:
            return bool(method())
        except Exception:
            return False
    return bool(method)


def _agent_query_str(agent: Any, method_name: str) -> Optional[str]:
    """Like _agent_query_bool but for string-returning methods (e.g. phase_name)."""
    rpc_methods = getattr(agent, "supported_remote_methods", None)
    if rpc_methods is not None:
        if method_name in rpc_methods:
            try:
                return str(getattr(agent, method_name)())
            except Exception:
                return None
        return None
    method = getattr(agent, method_name, None)
    if method is None:
        return None
    if callable(method):
        try:
            return str(method())
        except Exception:
            return None
    return str(method)


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
    obs_preprocess: Optional[ObsPreprocess] = None,
) -> Dict[str, Any]:
    """Poll agent.act() until the IK solver has fully converged.

    Convergence requires two conditions:
      1. All arm joints are non-zero (IK has started producing output).
      2. Consecutive readings are close (joints have stabilized).

    On first call the JAX JIT in pyroki can take several seconds to compile.
    """
    logger.info("Waiting for IK solver to warm up and converge...")
    obs_dict = obs.to_dict()
    if obs_preprocess is not None:
        obs_dict = obs_preprocess(obs_dict)
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
    final_obs_dict = obs.to_dict()
    if obs_preprocess is not None:
        final_obs_dict = obs_preprocess(final_obs_dict)
    return agent.act(final_obs_dict)


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


def _gello_header_probe(host: str, port: int, timeout_s: float = 4.0) -> bool:
    """True iff a gello_position_server is accepting and streaming.

    Reads the 4-byte joint-count header. A bare TCP connect is NOT enough:
    a wedged server (captive to a stale client) still completes handshakes
    into its listen backlog without ever accepting or sending.
    """
    try:
        s = socket.socket()
        s.settimeout(timeout_s)
        s.connect((host, port))
        buf = b""
        while len(buf) < 4:
            chunk = s.recv(4 - len(buf))
            if not chunk:
                return False
            buf += chunk
        s.close()
        return True
    except OSError:
        return False


def _preflight_gello_server(agent_cfg: Any) -> None:
    """For network-GELLO agents, verify the leader server is serving before
    anything else spins up — and auto-restart it when it isn't.

    The v1 server is single-client and can be held captive by a stale
    connection leaked by the host DLP's transparent proxy; every launch after
    such a leak would otherwise abort at the first agent action. Launch start
    is the one safe moment to restart automatically: the leader arms are
    expected to be racked in their rest slots, which is exactly what the
    server's zero-pose offset capture needs.
    """
    target = str(agent_cfg.get("_target_", ""))
    host = agent_cfg.get("host", None)
    if "YamGelloAgent" not in target or not host:
        return
    port = int(agent_cfg.get("network_port", 5555))
    if _gello_header_probe(host, port):
        logger.info(f"GELLO server at {host}:{port} is serving.")
        return
    logger.warning(
        f"GELLO server at {host}:{port} is not serving (down or wedged on a stale client) — "
        "restarting it automatically. Leader arms should be in their rest slots "
        "(zero-pose offsets are recaptured on server start)."
    )
    script = os.path.join(ROOT_PATH, "scripts", "start_gello_server.sh")
    try:
        result = subprocess.run(
            ["bash", script, str(host)],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=str(ROOT_PATH),
        )
        for line in (result.stdout or "").strip().splitlines()[-3:]:
            logger.info(f"  [gello launcher] {line}")
    except (subprocess.TimeoutExpired, OSError) as e:
        raise RuntimeError(f"GELLO server auto-restart failed to run: {e}") from e
    if not _gello_header_probe(host, port, timeout_s=6.0):
        raise RuntimeError(
            f"GELLO server at {host}:{port} is still not serving after an automatic restart. "
            f"Inspect it: ssh cat@{host} 'tail -30 ~/gello_server.log'"
        )
    logger.info("GELLO server restarted and serving.")


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

        if args.vlm:
            backend, model = _normalize_vlm(args.vlm)
            n = _override_vlm(configs_dict, backend, model)
            logger.info("VLM override: --vlm {} -> backend '{}'{} on {} detector block(s)",
                        args.vlm, backend, f" (model {model})" if model else "", n)

        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        api_servers = configs_dict.pop("api_servers", None)

        # Build a launch-side image preprocessor from the agent's obs_transform
        # config. When present, this shrinks per-tick Portal RPC payloads from
        # native ~2.7 MB to ~150 KB at typical 224x224 / 256x256 policy inputs.
        obs_preprocess = _build_obs_image_preprocess(agent_cfg)
        if obs_preprocess is not None:
            obs_xform_cfg = agent_cfg.get("obs_transform", {})
            logger.info(
                "Pre-RPC image resize enabled (image_size={}, transform={})",
                obs_xform_cfg.get("image_size"),
                obs_xform_cfg.get("_target_", "?").rsplit(".", 1)[-1],
            )

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
                _run_sim_control_loop(
                    robots, agent, main_config, display=display, obs_preprocess=obs_preprocess
                )
            finally:
                display.stop()
            return

        # ----- Real hardware mode (original path) ----- #
        _preflight_gello_server(agent_cfg)
        logger.info("Initializing sensors...")
        camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)

        setup_can_interfaces()

        logger.info("Initializing robots...")
        robots = initialize_robots(main_config.robots, server_processes)

        # Release any operator-backdriven robots (e.g. leader arms in a
        # bilateral / DAgger setup) before the control loop touches them.
        for name in main_config.release_at_startup:
            if name not in robots:
                logger.warning(f"release_at_startup: '{name}' is not in robots {list(robots.keys())}, skipping")
                continue
            try:
                if main_config.release_kd_scale > 0.0:
                    logger.info(
                        f"Releasing '{name}' (damped_compliant_mode, kd_scale={main_config.release_kd_scale:.3f}) "
                        f"per release_at_startup"
                    )
                    robots[name].damped_compliant_mode(main_config.release_kd_scale)
                else:
                    logger.info(f"Releasing '{name}' (zero_torque_mode) per release_at_startup")
                    robots[name].zero_torque_mode()
            except Exception as e:
                logger.warning(f"Could not release '{name}': {e}")

        agent = initialize_agent(agent_cfg, server_processes)

        # Create a standalone ViserMonitor for agents that don't have their own
        # (e.g. GELLO, VR).  YamViserAgent already embeds a ViserMonitor.
        monitor: Optional[ViserMonitor] = None
        agent_target = agent_cfg.get("_target_", "")
        if main_config.enable_monitor and "YamViserAgent" not in agent_target:
            is_bimanual = len(robots) > 1
            right_extrinsic = (
                main_config.station_metadata.get("extrinsics", {}).get("right_arm_extrinsic")
                if main_config.station_metadata
                else None
            )
            monitor = ViserMonitor(
                enable_urdf=True,
                bimanual=is_bimanual,
                right_arm_extrinsic=right_extrinsic,
            )
            logger.info("ViserMonitor started (standalone) for camera feeds + recording + URDF")

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
        # Save the pose of every arm at reset.  Followers will be commanded
        # back to their saved pose at shutdown via _safe_move_robots; released
        # robots (leaders) are *also* returned to their saved pose, after a
        # position_mode call that restores their PD gains.  Skipping
        # release_at_startup robots from this dict caused leaders to free-fall
        # during the shutdown soft_release ramp.
        saved_positions = _save_robot_positions(obs, list(robots.keys()))
        logger.info(f"Saved pre-teleop positions for: {list(saved_positions.keys())}")
        logger.info(f"Action spec: {env.action_spec()}")

        # Only IK-based agents (Viser, VR) need convergence warm-up.
        # Direct-joint agents (GELLO) produce valid actions immediately.
        _IK_AGENT_PATTERNS = ["YamViserAgent", "YamVrAgent"]
        agent_needs_ik = any(pat in agent_target for pat in _IK_AGENT_PATTERNS)

        if agent_needs_ik:
            initial_action = _wait_for_ik_convergence(
                agent, obs, list(robots.keys()), obs_preprocess=obs_preprocess
            )
        else:
            logger.info("Agent does not use IK, skipping convergence wait.")
            initial_obs = obs.to_dict()
            if obs_preprocess is not None:
                initial_obs = obs_preprocess(initial_obs)
            # Same guard as the control loop's act() call: a wedged portal RPC
            # (or a leader source that never delivers data) otherwise hangs the
            # launch here forever, silently re-sending multi-MB obs payloads.
            with Timeout(30, "Initial agent action"):
                initial_action = agent.act(initial_obs)

        # Apply any robot mode-switches the agent emitted on its first tick
        # (e.g. DAgger's leaders need position_mode before the safe-move so the
        # PD gains are restored — _safe_move_robots calls robot.move_joints
        # directly, bypassing env.step / _apply_action where modes normally land).
        for name, entry in initial_action.items():
            if not isinstance(entry, dict) or "mode" not in entry:
                continue
            robot = robots.get(name)
            if robot is None:
                continue
            mode = entry["mode"]
            try:
                if mode == "position":
                    logger.info(f"Initial mode for '{name}' -> position (restoring PD gains)")
                    robot.position_mode()
                elif mode == "zero_torque":
                    logger.info(f"Initial mode for '{name}' -> zero_torque")
                    robot.zero_torque_mode()
                elif mode == "compliant":
                    kd_scale = float(entry.get("kd_scale", 0.1))
                    logger.info(f"Initial mode for '{name}' -> compliant (kd_scale={kd_scale:.3f})")
                    robot.damped_compliant_mode(kd_scale)
                else:
                    logger.warning(f"Unknown initial mode '{mode}' for '{name}' — ignoring")
            except Exception as e:
                logger.warning(f"Could not apply initial mode '{mode}' to '{name}': {e}")

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
        resolved_robot_configs = _resolve_robot_configs(main_config.robots)
        if main_config.collection is not None:
            session = instantiate(main_config.collection)
            session.recorder.robot_configs = resolved_robot_configs
            logger.info("DataCollectionSession configured (target={} episodes)", session.num_episodes)
        elif main_config.recording is not None:
            recorder = instantiate(main_config.recording)
            recorder.robot_configs = resolved_robot_configs
            logger.info("EpisodeRecorder configured (base_dir={})", recorder.base_dir)

        display = StatusDisplay()
        display.start()
        if session is not None:
            session.display = display

        logger.info("Starting control loop...")
        try:
            _run_control_loop(
                env, agent, main_config,
                monitor=monitor, recorder=recorder, session=session, display=display,
                obs_preprocess=obs_preprocess,
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
                # Released robots are in zero_torque_mode (kp=kd=0).  Restore
                # PD gains before move-back so move_joints can actually drive
                # them; position_mode also seeds the command target with the
                # current pose so the PD doesn't lurch toward a stale (zero)
                # target.
                for name in main_config.release_at_startup:
                    if name not in robots:
                        continue
                    try:
                        logger.info(f"Restoring PD gains on '{name}' (position_mode) before park-pose move")
                        robots[name].position_mode()
                    except Exception as e:
                        logger.warning(f"position_mode failed for '{name}': {e}")

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
    obs_preprocess: Optional[ObsPreprocess] = None,
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

            obs_dict = obs.to_dict()
            if obs_preprocess is not None:
                obs_dict = obs_preprocess(obs_dict)
            action = agent.act(obs_dict)

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


# A correctly-calibrated YAM gripper reports normalized gripper_pos in [0, 1]
# (0 = closed, 1 = open). If a gripper motor re-zeros between sessions, the
# arm's hardcoded gripper_limits stop matching the hardware and gripper_pos
# drifts outside [0, 1] — which both freezes the gripper (commands get clamped
# past the stop) AND silently writes mis-normalized values into recordings.
# This guard surfaces that immediately instead of after a ruined dataset.
_GRIPPER_NORM_LO = -0.1
_GRIPPER_NORM_HI = 1.1
_GRIPPER_WARN_THROTTLE_S = 10.0


def _warn_on_stale_gripper_calibration(obs: Observation, last_warn: Dict[str, float]) -> None:
    """Warn (throttled per arm) when an arm's gripper_pos leaves [0, 1].

    Out-of-range normalized gripper readings are the unmistakable signature of
    stale ``gripper_limits`` after a gripper motor re-zeroed. Catching it at
    runtime turns a silent, dataset-corrupting failure into an obvious alert.
    """
    now = time.monotonic()
    for name, arm in obs.arms.items():
        gp = getattr(arm, "gripper_pos", None)
        if gp is None:
            continue
        val = float(np.asarray(gp).reshape(-1)[0])
        if _GRIPPER_NORM_LO <= val <= _GRIPPER_NORM_HI:
            continue
        if now - last_warn.get(name, 0.0) < _GRIPPER_WARN_THROTTLE_S:
            continue
        last_warn[name] = now
        logger.warning(
            "Arm '{}' gripper_pos={:.2f} is outside [0,1] — gripper_limits are likely STALE "
            "(gripper motor re-zeroed). Re-measure with "
            "`uv run scripts/diagnostics/test_gripper_range.py --channel <ch>` and update that arm's "
            "gripper_limits BEFORE recording, or this session's gripper data will be inconsistent.",
            name,
            val,
        )


def _run_control_loop(
    env: RobotEnv,
    agent: Agent,
    config: LaunchConfig,
    monitor: Optional[ViserMonitor] = None,
    recorder: Optional[EpisodeRecorder] = None,
    session: Optional[DataCollectionSession] = None,
    display: Optional[StatusDisplay] = None,
    obs_preprocess: Optional[ObsPreprocess] = None,
) -> None:
    """Run the main control loop.  Exits when _shutdown_requested is set by SIGINT.

    ``obs_preprocess`` is an optional callable that mutates / resizes the
    obs-dict in place before it's serialized for the agent's Portal RPC.
    See _build_obs_image_preprocess(): when set, the per-tick payload over
    Portal is shrunk by ~10-12x for the typical 3-camera policy obs.
    """
    steps = 0
    start_time = time.time()
    loop_count = 0
    slow_hz_streak = 0  # consecutive 1-second windows below SLOW_HZ_FRACTION * target
    # Track the agent's phase across ticks so we can log edges from the *main*
    # process (the TUI's Rich-aware sink) instead of relying on the agent
    # subprocess's stderr which can be clobbered by Live-panel redraws.
    last_phase_seen: Optional[str] = None
    # Label for phase-edge logs: agents may provide their own (e.g. the SubRL
    # online-RL loop -> "SubRL loop"); default keeps the DAgger wording.
    phase_log_label = _agent_query_str(agent, "phase_log_label") or "DAgger phase" 

    obs = env.reset()

    # Per-iteration timings for stages OUTSIDE env.step (which has its own
    # last_step_timing). Merged into the slow-Hz warning so unmeasured
    # stages — agent RPC, recorder, monitor — show up in the breakdown.
    extra_timing: Dict[str, float] = {}

    # Per-arm last-warn timestamps for the stale-gripper-calibration guard.
    gripper_warn_last: Dict[str, float] = {}

    while not _shutdown_requested:
        t_iter_start = time.perf_counter()

        # Loud alert if any gripper_pos has drifted outside [0,1] (stale
        # gripper_limits after a motor re-zero) — protects recorded data.
        _warn_on_stale_gripper_calibration(obs, gripper_warn_last)

        t0 = time.perf_counter()
        obs_dict = obs.to_dict()
        # PR #10: pre-resize obs images on this side of the Portal RPC so the
        # payload to the agent process is ~12x smaller for typical 3-cam policy
        # configs. No-op when the agent has no obs_transform.image_size.
        if obs_preprocess is not None:
            obs_dict = obs_preprocess(obs_dict)
        extra_timing["obs_to_dict"] = time.perf_counter() - t0

        with Timeout(30, "Agent action"):
            t0 = time.perf_counter()
            action = agent.act(obs_dict)
            extra_timing["agent.act"] = time.perf_counter() - t0

        # Data collection session manages recording + trigger signals.
        # `intervention` is True iff the agent reports it just emitted an
        # operator-correction action (DAgger).  Non-DAgger agents do not
        # expose is_correcting; the helper returns False and recordings
        # behave identically to before this flag existed.
        intervention = _agent_query_bool(agent, "is_correcting")
        phase = _agent_query_str(agent, "phase_name")
        # Edge-log phase changes from the main process so they go through the
        # TUI's Rich sink (subprocess stderr can be clobbered by Live redraws).
        if phase is not None and phase != last_phase_seen:
            if last_phase_seen is not None:
                logger.info("{}: {} -> {}", phase_log_label, last_phase_seen, phase)
            last_phase_seen = phase
            # Mirror to the TUI panel so the operator can see the current
            # phase even when no session is providing SessionState.
            if display is not None and session is None:
                display.update_phase(phase)
        t0 = time.perf_counter()
        if session is not None:
            if not session.step(obs, action, intervention=intervention, phase=phase):
                break  # session complete or quit signal
            extra_timing["session.step"] = time.perf_counter() - t0
        elif recorder is not None and recorder.is_recording:
            # Standalone recorder: record pre-step (s_t, a_t)
            recorder.record(obs, action, intervention=intervention)
            extra_timing["recorder.record"] = time.perf_counter() - t0

        t_step_start = time.perf_counter()
        obs = env.step(action)
        step_duration = time.perf_counter() - t_step_start

        if step_duration > SLOW_STEP_WARN_S:
            timing = getattr(env, "last_step_timing", {})
            top = sorted(timing.items(), key=lambda kv: -kv[1])[:5]
            breakdown = ", ".join(f"{k}={v * 1000:.0f}ms" for k, v in top) or "no timing data"
            logger.warning(
                f"Env step took {step_duration * 1000:.0f}ms (>{SLOW_STEP_WARN_S * 1000:.0f}ms): {breakdown}"
            )

        t0 = time.perf_counter()
        if monitor is not None:
            monitor.update(obs)
        extra_timing["monitor.update"] = time.perf_counter() - t0

        extra_timing["iter_total"] = time.perf_counter() - t_iter_start

        steps += 1
        loop_count += 1

        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            hz = loop_count / elapsed_time
            if display is not None:
                display.update_loop(hz, steps)
            target_hz = config.hz
            if target_hz and hz < target_hz * SLOW_HZ_FRACTION:
                slow_hz_streak += 1
                # Log on entry to slow state and every SLOW_HZ_REPEAT_S after,
                # not every second — and include the last step's breakdown so
                # the warning is actually actionable.
                if slow_hz_streak == SLOW_HZ_STREAK_S or (
                    slow_hz_streak > SLOW_HZ_STREAK_S and (slow_hz_streak - SLOW_HZ_STREAK_S) % SLOW_HZ_REPEAT_S == 0
                ):
                    timing = dict(getattr(env, "last_step_timing", {}))
                    timing.update(extra_timing)
                    iter_total = timing.pop("iter_total", None)
                    top = sorted(timing.items(), key=lambda kv: -kv[1])[:8]
                    breakdown = ", ".join(f"{k}={v * 1000:.1f}ms" for k, v in top) or "no timing data"
                    iter_str = f" iter_total={iter_total * 1000:.1f}ms," if iter_total is not None else ""
                    logger.warning(
                        f"Loop at {hz:.1f} Hz (target {target_hz:.1f} Hz) for {slow_hz_streak}s.{iter_str} "
                        f"Last step stages: {breakdown}"
                    )
            else:
                slow_hz_streak = 0
            start_time = time.time()
            loop_count = 0

        if config.max_steps is not None and steps >= config.max_steps:
            logger.info(f"Reached max steps ({config.max_steps}), stopping...")
            break

    if _shutdown_requested:
        logger.info("Shutdown flag detected, exiting control loop.")


if __name__ == "__main__":
    main(tyro.cli(Args))
