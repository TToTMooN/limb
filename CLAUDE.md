# limb — CLAUDE.md

## Project Goal

Minimal, high-frequency control stack for **YAM bimanual arms** (I2RT).
Primary use cases: teleoperation for data collection, and running VLA policies.

---

## Architecture

```
configs/                  # YAML launch configs (OmegaConf)
robot_configs/            # Per-arm hardware specs
limb/
  envs/
    launch.py             # Main entry point
    robot_env.py          # dm_env environment wrapper
    configs/
      loader.py           # YAML → dataclass
      instantiate.py      # _target_ → Python class
  agents/
    agent.py              # Agent protocol
    teleoperation/
      yam_viser_agent.py  # Viser web-UI teleop (keep)
      yam_gello_agent.py  # GELLO Dynamixel teleop (keep)
      yam_vr_agent.py     # Pico VR teleop (keep)
    policy_learning/
      async_pi0_agent.py  # π0 VLA policy client (keep)
      diffusion_policy_agent.py  # Diffusion policy client (keep)
  robots/
    robot.py              # Robot protocol
    yam_motor_chain_robot.py  # YAM CAN bus driver (keep)
    inverse_kinematics/
      yam_pink.py         # Pinocchio QP IK (keep)
      yam_pyroki.py       # JAX IK (keep)
    viser/
      viser_base.py       # IK + URDF visualization
      viser_monitor.py    # Camera feeds + recording
    utils.py              # Rate limiter, Timeout
  sensors/
    cameras/
      camera.py           # Camera protocol
      realsense_camera.py # Intel RealSense (keep)
      opencv_camera.py    # Generic webcam (keep)
      zed_camera.py       # Stereolabs ZED (keep)
      camera_utils.py     # Depth/point-cloud utils (keep)
  dynamixel/
    dynamixel_reader.py          # USB GELLO
    network_dynamixel_reader.py  # Network GELLO (R1 Lite)
  utils/
    launch_utils.py       # CAN setup, safe-move helpers
    portal_utils.py       # Portal RPC for multi-process
    xr_client.py          # XRoboToolkit VR client
    depth_utils.py        # Point cloud processing
scripts/                  # Standalone diagnostic scripts
dependencies/             # Git submodules: i2rt, XRoboToolkit
```

### Control loop flow

```
launch.py
  └─ RobotEnv.step() @ 100 Hz
       ├─ Agent.act(obs) → action
       │    (teleop: reads device pose/joints → IK → joint targets)
       │    (policy: sends obs to VLA server → returns joint targets)
       └─ Robot.command(action)
            └─ YamMotorChainRobot → CAN bus → DM motors
```

---

## What to Keep

| Component | Files | Reason |
|-----------|-------|--------|
| YAM robot driver | `robots/yam_motor_chain_robot.py` | Core hardware |
| IK solvers | `robots/inverse_kinematics/yam_pink.py`, `yam_pyroki.py` | Required for Cartesian teleop |
| Viser teleop | `agents/teleoperation/yam_viser_agent.py` | Web UI teleop |
| GELLO teleop | `agents/teleoperation/yam_gello_agent.py` | Dynamixel leader |
| VR teleop | `agents/teleoperation/yam_vr_agent.py` | Pico headset |
| VLA policies | `agents/policy_learning/async_pi0_agent.py`, `diffusion_policy_agent.py` | Policy deployment |
| Viser monitor | `robots/viser/viser_monitor.py` | Camera + URDF display |
| Camera drivers | `sensors/cameras/*.py` | Observation space |
| Point cloud utils | `utils/depth_utils.py`, `sensors/cameras/camera_utils.py` | Depth obs |
| Portal RPC | `utils/portal_utils.py` | Multi-process coordination |
| Config system | `envs/configs/loader.py`, `instantiate.py` | Launch infra |
| JoyCon gripper | `input_devices/joycon_gripper_reader.py` | Gripper control during GELLO teleop |
| Scripts | `scripts/test_*.py` | Hardware diagnostics |

## What to Remove (Refactor Targets)

| Component | Files | Reason |
|-----------|-------|--------|
| Franka driver | `robots/franka_osc.py`, `robots/robotiq_gripper.py` | Not YAM |
| Franka agents | `agents/teleoperation/franka_*.py`, `agents/client/franka_*.py` | Not YAM |
| Franka serving | `serving/serve_pyroki_*.py` | Franka-only |
| Franka IK | `robots/inverse_kinematics/franka_pyroki.py` | Not YAM |
| Franka configs | `configs/franka/`, `robot_configs/franka/` | Not YAM |
| MuJoCo sim | `robots/mujoco_sim_robot.py`, `mujoco/` | Sim only |
| MJLab sim | `robots/mjlab_sim_robot.py`, `robots/yam_pick_red_cube_sim_robot.py` | Sim only |
| Sim agents | `agents/teleoperation/gello_leader_agent.py`, `bilateral_leader_agent.py` | Sim-only patterns |
| Sim configs | `configs/yam/yam_gello_mujoco_sim.yaml`, `yam_gello_mjlab_sim.yaml`, `yam_gello_pick_red_cube_sim.yaml`, `yam_bilateral_mjlab_sim.yaml` | Sim only |

---

## Launch Commands

### Teleoperation

```bash
# Viser web-UI (browser at localhost:8080)
uv run limb/envs/launch.py --config_path configs/yam/yam_viser_bimanual.yaml

# GELLO leader arms (USB Dynamixel)
uv run limb/envs/launch.py --config_path configs/yam/yam_gello_bimanual.yaml

# VR headset (Pico, requires XRoboToolkit service running)
uv run limb/envs/launch.py --config_path configs/yam/yam_vr_bimanual.yaml
```

### VLA Policy Deployment

```bash
# π0 policy (requires pi0 server)
uv run limb/envs/launch.py --config_path configs/yam/yam_pi0_bimanual.yaml

# Diffusion policy (requires websocket server)
uv run limb/envs/launch.py --config_path configs/yam/yam_diffusion_bimanual.yaml
```

### GELLO Network Mode (R1 Lite remote)

```bash
# Start GELLO position server on R1 Lite (at 10.42.0.1)
bash scripts/start_gello_server.sh

# Kill remote server
bash scripts/start_gello_server.sh --kill
```

### Hardware Diagnostics

```bash
# Test cameras
uv run scripts/test_realsense_cameras.py

# Test GELLO input
uv run scripts/test_gello_input.py

# Test VR input
uv run scripts/test_vr_input.py
```

---

## Hardware Setup

### CAN Interface (YAM arms)

One-time udev rule to auto-bring-up CAN at 1 Mbps:
```bash
echo 'SUBSYSTEM=="net", KERNEL=="can*", ACTION=="add", RUN+="/sbin/ip link set %k up type can bitrate 1000000"' \
  | sudo tee /etc/udev/rules.d/99-can.rules
sudo udevadm control --reload && sudo udevadm trigger
```

Check interfaces are up:
```bash
ip link show | grep can
```

Expected: `can_follow_l` and `can_follow_r` (or as configured in robot_configs/).

### GELLO (Dynamixel)

- USB-to-serial dongle connects to Dynamixel servos
- Baud rate: 4 Mbps (`dynamixel_reader.py`)
- Network mode (R1 Lite): TCP server at `10.42.0.1:port`

### VR (Pico headset)

```bash
bash scripts/install_xrobotoolkit_sdk.sh
# Then open XRoboToolkit app on Pico headset
```

---

## Config System

Configs are YAML files loaded by OmegaConf. Every object uses `_target_` for dynamic instantiation:

```yaml
# configs/yam/yam_viser_bimanual.yaml
_target_: limb.envs.launch.LaunchConfig
hz: 100
cameras:
  wrist_left:
    _target_: limb.sensors.cameras.realsense_camera.RealSenseCamera
    serial_number: "XXXXXX"
robots:
  left:
    _target_: limb.robots.yam_motor_chain_robot.YamMotorChainRobot
    # ...
  right:
    _target_: limb.robots.yam_motor_chain_robot.YamMotorChainRobot
    # ...
agent:
  _target_: limb.agents.teleoperation.yam_viser_agent.YamViserAgent
  # ...
```

Robot configs (`robot_configs/yam/left.yaml`, `right.yaml`) specify:
- Motor chain (CAN IDs, motor types, interface name)
- PID gains (`kp`, `kd`)
- Joint limits (6 arms + 1 gripper)
- URDF/XML path

---

## IK Solvers

| Solver | Class | Library | Style | Use When |
|--------|-------|---------|-------|----------|
| Pink | `YamPink` | Pinocchio + QP | Differential (velocity) | Smooth trajectories, production |
| PyRoki | `YamPyroki` | JAX | Global (one-shot) | Fast startup, less smooth |

Both expose the same interface and support bimanual simultaneous solving.

---

## VLA Policy Integration

Policies run as external servers and are called via async clients:

- **π0** (`async_pi0_agent.py`): HTTP client to OpenPI server
  - Sends: `{images: {...}, state: joint_positions}`
  - Receives: `action_chunk` (sequence of joint targets)

- **Diffusion** (`diffusion_policy_agent.py`): WebSocket client
  - Sends observation dict
  - Receives action chunk

Both agents support action chunking and async inference to hide server latency.

---

## Development Conventions

- **Package manager**: `uv` (not pip)
- **Python**: 3.11 exactly
- **Linter**: `ruff` (line length 119, config in `pyproject.toml`)
- **Logging**: `loguru` (`from loguru import logger`)
- **Rate control**: `Rate` class from `robots/utils.py`
- **Multi-process RPC**: `portal` library with `@remote()` decorator
- **Config**: OmegaConf + custom `instantiate()` (not Hydra)

### Run linter

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
loguru           # Logging
```

Install:
```bash
uv sync
```

---

## Refactor Status

See [.claude/refactor-plan.md](.claude/refactor-plan.md) for the detailed plan to strip Franka and sim code.
