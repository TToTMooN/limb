# limb — CLAUDE.md

## Project Goal

Minimal, high-frequency control stack for **YAM bimanual arms** (I2RT).
Primary use cases: teleoperation for data collection, and running VLA policies.
limb is the "robot side" — any VLA framework (OpenPI, LeRobot, etc.) can plug into it.

---

## Architecture

```
configs/                  # YAML launch configs (OmegaConf)
robot_configs/            # Per-arm hardware specs (CAN IDs, PID gains, limits)
limb/
  envs/
    launch.py             # Main entry point + control loop
    robot_env.py          # dm_env environment wrapper
    configs/
      loader.py           # YAML → OmegaConf dict
      instantiate.py      # _target_ → Python class
  core/
    observation.py        # Typed observation dataclasses
  agents/
    agent.py              # Agent protocol (act, action_spec)
    teleoperation/
      yam_viser_agent.py  # Viser web-UI teleop
      yam_gello_agent.py  # GELLO Dynamixel teleop
      yam_vr_agent.py     # Pico VR teleop
    policy_learning/
      policy_client.py    # PolicyClient protocol + OpenPI/WebSocket clients
      transforms.py       # Obs/action transforms (YAML-configurable)
      action_chunk_manager.py  # Action chunk buffering + temporal smoothing
      policy_agent.py     # YamPolicyAgent — composes client+transforms+chunking
  robots/
    robot.py              # Robot protocol
    yam_motor_chain_robot.py  # YAM CAN bus driver
    inverse_kinematics/
      yam_pink.py         # Pinocchio QP IK
      yam_pyroki.py       # JAX IK
    utils.py              # Rate limiter, Timeout
  devices/
    dynamixel_reader.py          # USB GELLO reader
    network_dynamixel_reader.py  # Network GELLO / R1 Lite
    joycon_gripper_reader.py     # JoyCon gripper control
    xr_client.py                 # XRoboToolkit VR client
  visualization/
    viser_base.py         # IK + URDF visualization
    viser_monitor.py      # Camera feeds + video recording
  recording/
    episode_recorder.py   # Raw episode recording (states/actions/video)
    trigger.py            # Hands-free trigger signals (keyboard/foot pedal/VR)
    session.py            # Multi-episode data collection session manager
  sensors/
    cameras/
      camera.py           # Camera protocol
      realsense_camera.py # Intel RealSense
      opencv_camera.py    # Generic webcam
      zed_camera.py       # Stereolabs ZED
      camera_utils.py     # Image utils, obs extraction
  utils/
    launch_utils.py       # CAN setup, safe-move helpers
    portal_utils.py       # Portal RPC (@remote decorator)
    depth_utils.py        # Point cloud processing
scripts/                  # Standalone diagnostic scripts
docs/
  teleop.md               # Detailed teleop docs
  data_collection.md      # Data collection + episode recording docs
  policy_server_spec.md   # Spec for companion policy server repo
dependencies/             # Git submodules: i2rt, XRoboToolkit
```

### Control loop + process model

```
Process 1..N (Portal): Cameras, robots, agent  # separate processes for HW isolation
────────────────────────────────────────
Main process, main thread:
  └─ control loop @ 100 Hz
       ├─ Agent.act(obs) → action           # Portal RPC
       ├─ session.step(obs, action)          # trigger poll + recording
       │    or recorder.record(obs, action)  # direct recording mode
       ├─ RobotEnv.step(action)              # Portal RPC
       └─ ViserMonitor.update(obs)           # viser has own server thread
```

Episode save (~100-200ms) runs synchronously between episodes. Robot holds last commanded position.

### Observation format

```python
{
  "timestamp": float,
  "left": {"joint_pos": (6,), "joint_vel": (6,), "gripper_pos": (1,), "ee_pose": (7,)},
  "right": {"joint_pos": (6,), "joint_vel": (6,), "gripper_pos": (1,), "ee_pose": (7,)},
  "left_wrist_camera": {"images": {"rgb": (H,W,3)}, "timestamp": float},
}
```

### Action format

```python
{"left": {"pos": (7,)}, "right": {"pos": (7,)}}  # 6 joints + 1 gripper
```

---

## Launch Commands

```bash
# Teleoperation
uv run limb/envs/launch.py --config_path configs/yam_viser_bimanual.yaml
uv run limb/envs/launch.py --config_path configs/yam_gello_bimanual.yaml
uv run limb/envs/launch.py --config_path configs/yam_vr_bimanual.yaml

# Data collection (overlay configs — combine with any teleop config)
uv run limb/envs/launch.py --config_path configs/yam_gello_network_bimanual.yaml configs/collection.yaml
uv run limb/envs/launch.py --config_path configs/yam_vr_bimanual.yaml configs/collection_vr.yaml

# Policy deployment
uv run limb/envs/launch.py --config_path configs/yam_pi0_bimanual.yaml
uv run limb/envs/launch.py --config_path configs/yam_policy_bimanual.yaml

# Diagnostics
uv run scripts/test_realsense_cameras.py
uv run scripts/test_gello_input.py
uv run scripts/test_vr_input.py
```

---

## Config System

YAML files with `_target_` for dynamic instantiation (Hydra-like, but custom `instantiate()`):

```yaml
_target_: limb.envs.launch.LaunchConfig
hz: 100.0
robots:
  left: ["robot_configs/yam/left.yaml"]
  right: ["robot_configs/yam/left.yaml", "robot_configs/yam/right.yaml"]
agent:
  _target_: limb.agents.teleoperation.yam_gello_agent.YamGelloAgent
collection:  # or recording: for standalone mode
  _target_: limb.recording.session.DataCollectionSession
  num_episodes: 10
  trigger:
    _target_: limb.recording.trigger.KeyboardTrigger
```

Robot configs (`robot_configs/yam/left.yaml`, `right.yaml`) specify motor chain (CAN IDs, motor types), PID gains, joint limits, URDF path.

---

## Development Conventions

- **Package manager**: `uv` (not pip). Run everything with `uv run`.
- **Python**: 3.11 exactly
- **CLI args**: `tyro` — use `tyro.cli(Args)` for script entry points (not argparse/click)
- **Logging**: `loguru` everywhere — `from loguru import logger`. Never use `print()` or `logging`.
- **Linter**: `ruff` (line length 119, config in pyproject.toml)
- **Config**: OmegaConf + custom `instantiate()` (not Hydra). `_target_` pattern for dynamic class creation.
- **Multi-process RPC**: `portal` library with `@remote()` decorator
- **Rate control**: `Rate` class from `robots/utils.py`
- **Dataclasses**: All configurable components are `@dataclass` for `_target_` instantiation
- **Lazy imports**: Optional deps (openpi_client, websockets, evdev) imported inside methods

### Lint

```bash
uv run ruff check limb/
uv run ruff format limb/
```

---

## Key Dependencies

```
i2rt             # YAM motor driver (CAN, local submodule)
viser==1.0.16    # 3D web visualization
pin-pink         # Pinocchio QP IK
pyroki           # JAX IK (git)
portal           # Multi-process RPC
dm-env==1.6      # Environment API
omegaconf        # Config loading
pyrealsense2     # Intel RealSense
xdof_sdk         # XRoboToolkit VR
dynamixel-sdk    # GELLO input device
websockets       # Policy server client
msgpack          # Wire serialization
loguru           # Logging
tyro             # CLI argument parsing
```

Install: `uv sync`
