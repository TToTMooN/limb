# limb

**Lightweight arm control for robot learning.**

limb is a minimal, ROS-free Python stack for high-frequency control of direct-drive robot arms. It handles teleoperation (web UI, GELLO leader arms, VR), camera streaming, and learned policy deployment — everything between bare metal and your research code.

Built for [I2RT YAM](https://github.com/i2rt-robotics/i2rt) bimanual arms. Designed to be extended to any CAN-bus arm (ARX, Agilex, etc.).

## Why limb?

- **No ROS.** Direct CAN bus at 100 Hz. One Python process, one config file, one launch command.
- **Plug-and-play teleop.** Viser web UI, GELLO Dynamixel leader arms (USB or network), Pico VR — swap with a YAML change.
- **Policy-ready.** Ship your VLA (pi0, Diffusion Policy) as a server, point limb at it, collect data or run inference.
- **Typed observations.** Structured `Observation` dataclasses flow through the main process; agents receive plain dicts over RPC — no magic keys to memorize.

## Quick start

```bash
git clone --recurse-submodules <your-repo-url>
cd limb

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv --python 3.11
uv sync
```

### One-time CAN setup (YAM arms)

```bash
echo 'SUBSYSTEM=="net", KERNEL=="can*", ACTION=="add", RUN+="/sbin/ip link set %k up type can bitrate 1000000"' \
  | sudo tee /etc/udev/rules.d/99-can.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Verify with `ip link show | grep can` — you should see `can_follow_l` and `can_follow_r`.

### Launch

```bash
# Viser web-UI teleop (open browser at localhost:8080)
uv run limb/envs/launch.py --config_path configs/yam/yam_viser_bimanual.yaml

# GELLO leader arms (USB Dynamixel)
uv run limb/envs/launch.py --config_path configs/yam_gello_bimanual.yaml

# GELLO over network (R1 Lite Teleop at 10.42.0.1)
uv run limb/envs/launch.py --config_path configs/yam_gello_network_bimanual.yaml

# Pico VR headset
uv run limb/envs/launch.py --config_path configs/yam_vr_bimanual.yaml
```

Every launch command follows the same pattern: one entrypoint, one YAML config.

## Teleoperation modes

### Viser (web browser)

The default. Opens a 3D scene with URDF visualization, camera feeds, and interactive handles for Cartesian end-effector control. IK is solved via [Pink](https://github.com/stephane-caron/pink) (Pinocchio QP) or [pyroki](https://github.com/chungmin99/pyroki) (JAX).

```bash
uv run limb/envs/launch.py --config_path configs/yam/yam_viser_bimanual.yaml
```

Single-arm mode is also available:

```bash
uv run limb/envs/launch.py --config_path configs/yam/yam_viser_single_arm.yaml
```

### GELLO (Dynamixel leader arms)

Direct joint-to-joint mapping — no IK needed. The leader and follower share the same kinematics so joint angles transfer directly.

**USB (direct):**
```bash
uv run limb/envs/launch.py --config_path configs/yam_gello_bimanual.yaml
```

**Network (R1 Lite Teleop over Ethernet):**

Start the position server on the remote device first:
```bash
bash scripts/start_gello_server.sh              # default host 10.42.0.1
bash scripts/start_gello_server.sh 10.42.0.2    # custom host
bash scripts/start_gello_server.sh --kill        # stop remote server
```

Then launch:
```bash
uv run limb/envs/launch.py --config_path configs/yam_gello_network_bimanual.yaml
```

### VR (Pico headset)

Cartesian control via Pico VR controllers through [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service).

**Setup (one-time):**
1. Install the [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) on your PC.
2. Install the SDK: `bash scripts/install_xrobotoolkit_sdk.sh`
3. Open the XRoboToolkit app on the Pico headset and verify poses stream in the PC Service.

**Launch:**
```bash
uv run limb/envs/launch.py --config_path configs/yam_vr_bimanual.yaml
```

**Controls:**

| Button | Function |
|---|---|
| **Grip** (hold) | Activate arm control — arm follows hand |
| **Grip** (release) | Freeze arm — re-grip to continue from new position |
| **Trigger** | Gripper (0 = open, fully pressed = closed) |

The Viser web UI stays active during VR teleop for monitoring.

## Policy deployment

Learned policies run as external servers. limb calls them as async clients inside a standard agent.

**pi0 (OpenPI):**
```bash
uv run limb/envs/launch.py --config_path configs/yam/yam_pi0_bimanual.yaml
```
Sends `{images, state}` over HTTP, receives action chunks.

**Diffusion Policy:**
```bash
uv run limb/envs/launch.py --config_path configs/yam/yam_diffusion_bimanual.yaml
```
Sends observations over WebSocket, receives action chunks.

Both support action chunking and async inference to hide server latency.

## Hardware diagnostics

```bash
# Test cameras (interactive picker, recording, depth)
uv run scripts/test_realsense_cameras.py
uv run scripts/test_realsense_cameras.py --all --depth

# Test GELLO input (USB or network)
uv run scripts/test_gello_input.py
uv run scripts/test_gello_input.py --host 10.42.0.1

# Test VR input
uv run scripts/test_vr_input.py
uv run scripts/test_vr_input.py --with-ik    # with IK + Viser visualization
```

## Architecture

```
launch.py
  └─ RobotEnv.step() @ 100 Hz
       ├─ Agent.act(obs_dict) → action        # runs in subprocess via portal RPC
       │    teleop: reads device → IK → joint targets
       │    policy: sends obs to server → action chunk
       └─ Robot.command(action)
            └─ YamMotorChainRobot → CAN bus → DM motors
```

Observations flow as typed `Observation` dataclasses in the main process and are converted to plain dicts at the portal RPC boundary (`obs.to_dict()`). This keeps agents portable while giving monitors and the control loop structured access.

```
limb/
  core/                    # Observation dataclasses
  envs/
    launch.py              # Main entry point
    robot_env.py           # dm_env wrapper, builds Observation from RPC results
    configs/               # YAML → dataclass loader + instantiate()
  agents/
    teleoperation/         # Viser, GELLO, VR agents
    policy_learning/       # pi0, Diffusion Policy agents
  robots/
    yam_motor_chain_robot.py  # CAN bus driver
    inverse_kinematics/       # Pink (Pinocchio) and pyroki (JAX) IK
    viser/                    # URDF viz + camera monitor
  sensors/cameras/         # RealSense, OpenCV, ZED drivers
  dynamixel/               # GELLO USB + network readers
  utils/                   # Rate control, portal RPC, VR client
configs/                   # YAML launch configs
robot_configs/             # Per-arm hardware specs (CAN IDs, PID gains, joint limits)
scripts/                   # Diagnostics and server scripts
dependencies/              # Git submodules (i2rt, XRoboToolkit)
```

## Extending limb

### Add a new agent

```python
from limb.agents.agent import Agent

class MyAgent(Agent):
    def act(self, obs):
        # obs is a plain dict: {"left": {"joint_pos": ..., ...}, "cam_name": {"images": ...}, ...}
        return {"left": {"pos": target_joints}, "right": {"pos": target_joints}}
```

Point your YAML config at it:
```yaml
agent:
  _target_: my_package.MyAgent
  my_param: 42
```

### Add a new robot arm

Implement the `Robot` protocol (`robots/robot.py`) with `get_observations()`, `command_joint_pos()`, etc. The CAN-bus driver pattern in `yam_motor_chain_robot.py` is a good starting point for any direct-drive arm.

## Development

```bash
uv run ruff check limb/     # lint
uv run ruff check --fix limb/  # auto-fix
uv run ruff format limb/    # format
```

Python 3.11 | Package manager: `uv` | Linter: `ruff` (line length 119)

## Acknowledgments

Some structure is inspired by [GELLO](https://github.com/wuphilipp/gello_software) and [robots_realtime](https://github.com/uynitsuj/robots_realtime).
