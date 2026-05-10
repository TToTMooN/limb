"""Policy dry-run diagnostic — query the policy server with real observations
WITHOUT commanding the robot.

Useful when running ``limb teleop --config-path configs/yam_policy_bimanual.yaml``
causes the robot to shake or behave erratically: this script spins up the same
cameras, robots (for reading only), and agent, then logs the action chunks the
policy returns and compares them against the current robot state and against a
reference recorded episode.

Usage::

    uv run scripts/diagnostics/dry_run_policy.py \\
        --config-path configs/yam_policy_bimanual.yaml \\
        --duration-s 10 \\
        --reference-episode recordings/pick_up_the_grey_cube_and_hand_it_to_another_hand_20260414_200508/episode_20260414_200600_0002

The robot is NEVER commanded — joints stay in their saved pre-run position.
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tyro
from loguru import logger

from limb.envs.configs.instantiate import instantiate
from limb.envs.configs.loader import DictLoader
from limb.envs.robot_env import RobotEnv
from limb.robots.utils import Rate
from limb.utils.launch_utils import (
    cleanup_processes,
    initialize_agent,
    initialize_robots,
    initialize_sensors,
    setup_can_interfaces,
    setup_logging,
)

_shutdown_requested = False


def _sigint_handler(signum: int, frame: Any) -> None:
    global _shutdown_requested
    _shutdown_requested = True


@dataclass
class Args:
    config_path: Tuple[str, ...] = ("configs/yam_policy_bimanual.yaml",)
    duration_s: float = 10.0
    hz: float = 10.0  # how often we query the policy (deliberately slow for clarity)
    reference_episode: Optional[str] = None
    log_level: str = "INFO"
    save_path: Optional[str] = None  # optional: save full action trace as .npz


def _flatten_action(action: Dict[str, Dict[str, np.ndarray]], arm_names: List[str]) -> np.ndarray:
    """Concat per-arm action.pos in the order specified by arm_names."""
    parts = []
    for name in arm_names:
        parts.append(np.asarray(action[name]["pos"]))
    return np.concatenate(parts)


def _flatten_state(obs_dict: Dict[str, Any], arm_names: List[str]) -> np.ndarray:
    """Concat [joint_pos, gripper_pos] per arm — same convention as training."""
    parts = []
    for name in arm_names:
        arm = obs_dict[name]
        jp = np.asarray(arm["joint_pos"]).reshape(-1)
        gp = np.asarray(arm.get("gripper_pos", [])).reshape(-1)
        parts.append(jp)
        if gp.size > 0:
            parts.append(gp)
    return np.concatenate(parts)


def _summarize(label: str, arr: np.ndarray, names: Optional[List[str]] = None) -> None:
    """Pretty-print per-dim min/max/mean for a (T, D) trajectory."""
    if arr.size == 0:
        logger.info("{}: <empty>", label)
        return
    logger.info("{} shape={} dtype={}", label, arr.shape, arr.dtype)
    _, D = arr.shape
    for i in range(D):
        col = arr[:, i]
        n = names[i] if names and i < len(names) else f"dim_{i}"
        logger.info(
            "  {:<22s}  min={:+.4f}  max={:+.4f}  mean={:+.4f}  std={:.4f}  first={:+.4f}  last={:+.4f}",
            n, col.min(), col.max(), col.mean(), col.std(), col[0], col[-1],
        )


def _summarize_deltas(label: str, arr: np.ndarray, names: Optional[List[str]] = None) -> None:
    """Per-step absolute deltas (consecutive frames). High deltas → shake."""
    if arr.shape[0] < 2:
        logger.info("{}: too short for deltas", label)
        return
    d = np.abs(np.diff(arr, axis=0))
    logger.info("{} — per-step |delta| (T-1={})", label, d.shape[0])
    for i in range(d.shape[1]):
        n = names[i] if names and i < len(names) else f"dim_{i}"
        logger.info(
            "  {:<22s}  max={:.4f}  mean={:.4f}  std={:.4f}",
            n, d[:, i].max(), d[:, i].mean(), d[:, i].std(),
        )


def _load_reference(ep_dir: Path) -> Optional[Dict[str, Any]]:
    if not ep_dir.exists():
        logger.warning("Reference episode not found: {}", ep_dir)
        return None
    arm_files = sorted(ep_dir.glob("*_actions.npz"))
    if not arm_files:
        logger.warning("Reference episode has no *_actions.npz: {}", ep_dir)
        return None
    arms: Dict[str, Dict[str, np.ndarray]] = {}
    for f in arm_files:
        arm = f.stem.replace("_actions", "")
        arms[arm] = dict(np.load(str(f)))
        states_path = ep_dir / f"{arm}_states.npz"
        if states_path.exists():
            arms[arm]["__state"] = dict(np.load(str(states_path)))  # type: ignore[assignment]
    md_path = ep_dir / "metadata.json"
    md = json.loads(md_path.read_text()) if md_path.exists() else {}
    return {"arms": arms, "metadata": md}


def main(args: Args) -> None:
    """Spin up cameras + robots + policy agent, query policy in a loop, never move the robot."""
    setup_logging(level=args.log_level)
    logger.info("=" * 70)
    logger.info("POLICY DRY-RUN DIAGNOSTIC — robot is NOT commanded")
    logger.info("=" * 70)

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    server_processes: list = []
    agent: Any = None
    env: Optional[RobotEnv] = None

    try:
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])
        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        configs_dict.pop("api_servers", None)
        configs_dict.pop("collection", None)
        configs_dict.pop("recording", None)
        main_config = instantiate(configs_dict)

        logger.info("Initializing sensors...")
        camera_dict, _ = initialize_sensors(sensors_cfg, server_processes)

        setup_can_interfaces()

        logger.info("Initializing robots (read-only)...")
        robots = initialize_robots(main_config.robots, server_processes)
        arm_names = list(robots.keys())

        agent = initialize_agent(agent_cfg, server_processes)

        env = RobotEnv(
            robot_dict=robots,
            camera_dict=camera_dict,
            control_rate_hz=Rate(args.hz, rate_name="dry_run"),
        )

        # Show what the agent advertises
        logger.info("Agent action_spec: {}", agent.action_spec() if hasattr(agent, "action_spec") else "<n/a>")

        # Inspect the obs transform — useful for verifying state/image keys
        ot = getattr(agent_cfg, "obs_transform", None) or agent_cfg.get("obs_transform", {})
        if isinstance(ot, dict):
            logger.info("ObsTransform state_keys: {}", ot.get("state_keys"))
            logger.info("ObsTransform image_keys: {}", ot.get("image_keys"))
            logger.info("ObsTransform prompt:    {!r}", ot.get("prompt"))
            logger.info("ObsTransform image_size: {}", ot.get("image_size"))
        at = agent_cfg.get("action_transform", {})
        if isinstance(at, dict):
            logger.info("ActionTransform arm_names: {}", at.get("arm_names"))
            logger.info("ActionTransform joints_per_arm: {}", at.get("joints_per_arm"))
            logger.info("ActionTransform gripper_clip: {}", at.get("gripper_clip"))
        logger.info("Agent action_horizon: {}, smoothing_window: {}", agent_cfg.get("action_horizon"), agent_cfg.get("smoothing_window"))

        # Warm up: build one observation
        obs = env.reset()
        obs_dict = obs.to_dict()
        logger.info("First obs keys: {}", list(obs_dict.keys()))
        for name in arm_names:
            arm = obs_dict[name]
            logger.info(
                "  arm '{}': joint_pos={}, gripper_pos={}",
                name, np.asarray(arm["joint_pos"]).round(4), np.asarray(arm["gripper_pos"]).round(4) if "gripper_pos" in arm else None,
            )

        # ----- Main loop: query policy, log, DO NOT MOVE ROBOT ----- #
        n_total_steps = int(args.duration_s * args.hz)
        logger.info("Querying policy for ~{:.1f}s at {} Hz ({} steps)...", args.duration_s, args.hz, n_total_steps)

        action_log: List[np.ndarray] = []
        state_log: List[np.ndarray] = []
        timing_log: List[float] = []

        rate = Rate(args.hz, rate_name="dry_run_loop")
        start = time.time()
        step = 0
        while not _shutdown_requested and step < n_total_steps:
            obs = env.get_obs()
            obs_dict = obs.to_dict()
            t0 = time.time()
            action = agent.act(obs_dict)
            t1 = time.time()

            flat_act = _flatten_action(action, arm_names)
            flat_state = _flatten_state(obs_dict, arm_names)
            action_log.append(flat_act)
            state_log.append(flat_state)
            timing_log.append(t1 - t0)

            # Print compact per-step line
            if step < 5 or step % max(1, int(args.hz)) == 0:
                logger.info(
                    "step {:>4d}  act.dt={:.3f}s  act[0:3]={}  state[0:3]={}",
                    step, t1 - t0,
                    np.round(flat_act[:3], 3).tolist(),
                    np.round(flat_state[:3], 3).tolist(),
                )

            step += 1
            rate.sleep()

        elapsed = time.time() - start
        logger.info("Loop done: {} steps in {:.2f}s (effective {:.1f} Hz)", step, elapsed, step / max(elapsed, 1e-6))

        if not action_log:
            logger.warning("No actions logged.")
            return

        actions = np.stack(action_log)
        states = np.stack(state_log)
        timings = np.array(timing_log)

        # Build readable names: [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]
        action_names: List[str] = []
        for name in arm_names:
            for j in range(6):
                action_names.append(f"{name}_joint_{j}")
            action_names.append(f"{name}_gripper")
        state_names = action_names[:]  # state has identical layout when gripper_pos is present

        logger.info("")
        logger.info("=" * 70)
        logger.info("ACTIONS returned by policy (this run, dry — not commanded)")
        logger.info("=" * 70)
        _summarize("ACTIONS", actions, action_names)
        logger.info("")
        _summarize_deltas("ACTIONS Δ/step", actions, action_names)

        logger.info("")
        logger.info("=" * 70)
        logger.info("CURRENT ROBOT STATE (read during dry run)")
        logger.info("=" * 70)
        _summarize("STATE", states, state_names)

        logger.info("")
        logger.info("=" * 70)
        logger.info("STATE → ACTION GAP per dim (mean over the run)")
        logger.info("=" * 70)
        gap = actions - states
        for i in range(actions.shape[1]):
            n = action_names[i]
            logger.info(
                "  {:<22s}  mean(act-state)={:+.4f}  max|act-state|={:.4f}",
                n, gap[:, i].mean(), np.abs(gap[:, i]).max(),
            )
        logger.info("")
        logger.info(
            "Policy inference latency: mean={:.3f}s  p95={:.3f}s  max={:.3f}s",
            timings.mean(), np.quantile(timings, 0.95), timings.max(),
        )

        # ----- Optional comparison with a reference recorded episode ----- #
        if args.reference_episode:
            ref = _load_reference(Path(args.reference_episode))
            if ref:
                logger.info("")
                logger.info("=" * 70)
                logger.info("REFERENCE TRAINING EPISODE: {}", args.reference_episode)
                logger.info("=" * 70)
                for arm in arm_names:
                    if arm not in ref["arms"]:
                        logger.warning("Arm '{}' missing in reference episode", arm)
                        continue
                    ref_act = ref["arms"][arm]["pos"]
                    logger.info("REF [{}] action.pos shape={}", arm, ref_act.shape)
                    for j in range(ref_act.shape[1]):
                        col = ref_act[:, j]
                        logger.info(
                            "  {:<10s}_dim_{}  min={:+.4f} max={:+.4f} mean={:+.4f} std={:.4f}",
                            arm, j, col.min(), col.max(), col.mean(), col.std(),
                        )
                    ref_state_dict = ref["arms"][arm].get("__state")
                    if ref_state_dict is not None and "gripper_pos" in ref_state_dict:
                        gp = ref_state_dict["gripper_pos"]
                        logger.info(
                            "  {} state.gripper_pos: min={:+.4f} max={:+.4f}",
                            arm, gp.min(), gp.max(),
                        )

        # ----- Heuristic checks ----- #
        logger.info("")
        logger.info("=" * 70)
        logger.info("HEURISTIC CHECKS")
        logger.info("=" * 70)
        n_warn = 0
        # 1. Per-step delta sanity (at args.hz, max delta in rad/step)
        max_step_delta = np.abs(np.diff(actions, axis=0)).max() if actions.shape[0] > 1 else 0.0
        rad_per_s = max_step_delta * args.hz
        logger.info("Max per-step action delta = {:.3f} rad → ~{:.1f} rad/s at {} Hz", max_step_delta, rad_per_s, args.hz)
        if rad_per_s > 10.0:
            logger.warning("  ⚠ Action stream changes faster than 10 rad/s — robot will shake under PD control.")
            n_warn += 1
        # 2. Gripper range
        for i, n in enumerate(action_names):
            if n.endswith("gripper"):
                col = actions[:, i]
                logger.info("{} commanded range: [{:+.3f}, {:+.3f}]", n, col.min(), col.max())
                if col.max() <= 1.001 and col.min() >= -0.001:
                    logger.warning(
                        "  ⚠ {} stays in [0,1]. Training-action gripper goes up to 2.4 — "
                        "ActionTransform.gripper_clip in YAML may be wrong (should be [0.0, 2.4]).", n,
                    )
                    n_warn += 1
        # 3. State vector dimension vs training (14 for bimanual YAM)
        if states.shape[1] != 14:
            logger.warning("  ⚠ State dim = {} (expected 14 for bimanual). Check state_keys order.", states.shape[1])
            n_warn += 1
        # 4. Latency budget
        if timings.mean() * args.hz > 1.0:
            logger.warning(
                "  ⚠ Inference latency mean={:.3f}s exceeds 1/hz={:.3f}s — async chunks will be replaced before consumption.",
                timings.mean(), 1.0 / args.hz,
            )
            n_warn += 1
        if n_warn == 0:
            logger.info("  ✓ No obvious red flags detected.")

        # Save trace
        if args.save_path:
            out = Path(args.save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(out),
                actions=actions,
                states=states,
                inference_dt_s=timings,
                action_names=np.array(action_names),
                state_names=np.array(state_names),
            )
            logger.info("Trace saved: {}", out)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        logger.info("Cleaning up (robot was never commanded)...")
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if agent is not None:
            cleanup_processes(agent, server_processes)
        else:
            cleanup_processes(None, server_processes)
        signal.signal(signal.SIGINT, original_sigint)


if __name__ == "__main__":
    main(tyro.cli(Args))
