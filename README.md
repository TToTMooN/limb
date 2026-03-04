# limb

**Lightweight arm control for robot learning.**

Minimal, ROS-free Python stack for high-frequency control of direct-drive robot arms.
Teleoperation, camera streaming, and learned policy deployment — everything between bare metal and your research code.

Built for [I2RT YAM](https://github.com/i2rt-robotics/i2rt) bimanual arms.

---

**Contents:**
[Install](#install) ·
[Hardware Setup](#hardware-setup) ·
[Teleoperation](#teleoperation) ·
[Policy Deployment](#policy-deployment) ·
[Diagnostics](#hardware-diagnostics) ·
[Architecture](#architecture) ·
[Extending](#extending-limb) ·
[Development](#development)

---

## Why limb?


|                          |                                                                              |
| ------------------------ | ---------------------------------------------------------------------------- |
| **No ROS**               | Direct CAN bus at 100 Hz. One process, one config, one launch command.       |
| **Plug-and-play teleop** | Viser web UI, GELLO leader arms, Pico VR — swap with a YAML change.          |
| **Policy-ready**         | Ship your VLA as a server, point limb at it, run inference.                  |
| **Typed observations**   | Structured `Observation` dataclasses in the main loop; plain dicts over RPC. |


---

## Install

```bash
git clone --recurse-submodules https://github.com/TToTMooN/limb
cd limb

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv --python 3.11
uv sync
```

> **Submodule:** The repo includes [i2rt](https://github.com/i2rt-robotics/i2rt) (motor driver) as a git submodule under `dependencies/`. Make sure you clone with `--recurse-submodules`. The XRoboToolkit SDK for VR is installed separately — see [VR setup](#vr-pico-headset).

---

## Hardware Setup

### CAN Interface (YAM arms)

One-time udev rule to auto-bring-up CAN at 1 Mbps:

```bash
echo 'SUBSYSTEM=="net", KERNEL=="can*", ACTION=="add", RUN+="/sbin/ip link set %k up type can bitrate 1000000"' \
  | sudo tee /etc/udev/rules.d/99-can.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Verify: `ip link show | grep can` — expect `can_follow_l` and `can_follow_r`.

By default CAN interfaces show up as `can0`, `can1`, etc. The robot configs expect named interfaces (`can_follow_l`, `can_follow_r`). To rename them, add a udev rule:

```bash
# Find each adapter's device path
udevadm info -a /sys/class/net/can0 | grep serial

# Then create a rule mapping serial → name, e.g.:
echo 'SUBSYSTEM=="net", KERNEL=="can*", ATTRS{serial}=="XXXXXXXX", NAME="can_follow_l"' \
  | sudo tee -a /etc/udev/rules.d/99-can.rules
```

Alternatively, edit `channel` in `robot_configs/yam/left.yaml` / `right.yaml` to match your interface names directly.

### GELLO (Dynamixel leader arms)

- **USB mode:** Plug the USB-to-serial dongle into the host. Baud: 4 Mbps. *(Not yet verified — network mode is recommended.)*
- **Network mode (R1 Lite):** TCP server at `10.42.0.1`. See [GELLO Network](#gello-network-mode-r1-lite) below.

### VR (Pico headset)

1. Install the [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) on your workstation.
2. Install the Python SDK:
  ```bash
   bash scripts/install_xrobotoolkit_sdk.sh
  ```
3. Open the **XRoboToolkit** app on the Pico headset and confirm poses stream in the PC Service.

---

## Teleoperation

Every launch follows the same pattern — one entrypoint, one YAML config:

```bash
uv run limb/envs/launch.py --config_path <config.yaml>
```

### Viser (web browser)

3D scene with URDF visualization, camera feeds, and interactive Cartesian handles.
IK via [Pink](https://github.com/stephane-caron/pink) (Pinocchio QP) or [pyroki](https://github.com/chungmin99/pyroki) (JAX).

```bash
# Bimanual (open browser at localhost:8080)
uv run limb/envs/launch.py --config_path configs/yam_viser_bimanual.yaml

# Single arm
uv run limb/envs/launch.py --config_path configs/yam_viser_single_arm.yaml
```

### GELLO (Dynamixel leader arms)

Direct joint-to-joint mapping — leader and follower share kinematics, no IK needed.

> **Note:** USB mode is not yet fully verified. Network mode (R1 Lite) is the recommended setup.

```bash
# USB (direct connection)
uv run limb/envs/launch.py --config_path configs/yam_gello_bimanual.yaml
```

#### GELLO Network Mode (R1 Lite)

Start the position server on the remote device, then launch locally:

```bash
# On R1 Lite
bash scripts/start_gello_server.sh              # default 10.42.0.1
bash scripts/start_gello_server.sh 10.42.0.2    # custom host
bash scripts/start_gello_server.sh --kill        # stop server

# On workstation
uv run limb/envs/launch.py --config_path configs/yam_gello_network_bimanual.yaml
```

### VR (Pico headset)

Cartesian control via Pico VR controllers through [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service). Requires the [VR setup](#vr-pico-headset) above.

```bash
uv run limb/envs/launch.py --config_path configs/yam_vr_bimanual.yaml
```


| Button             | Function                                           |
| ------------------ | -------------------------------------------------- |
| **Grip** (hold)    | Activate arm — arm follows hand                    |
| **Grip** (release) | Freeze arm — re-grip to continue from new position |
| **Trigger**        | Gripper (0 = open, fully pressed = closed)         |


The Viser web UI stays active during VR teleop for monitoring.

---

## Policy Deployment

Learned policies run as external servers. limb calls them as async clients inside a standard agent. Both support action chunking and async inference to hide server latency.

### pi0 (OpenPI)

Sends `{images, state}` over HTTP, receives action chunks.

```bash
uv run limb/envs/launch.py --config_path configs/yam_pi0_bimanual.yaml
```

### Diffusion Policy

Sends observations over WebSocket, receives action chunks.

```bash
uv run limb/envs/launch.py --config_path configs/yam_diffusion_bimanual.yaml
```

---

## Hardware Diagnostics

```bash
# Cameras (interactive picker, recording, depth)
uv run scripts/test_realsense_cameras.py
uv run scripts/test_realsense_cameras.py --all --depth

# GELLO input (USB or network)
uv run scripts/test_gello_input.py
uv run scripts/test_gello_input.py --host 10.42.0.1

# VR input
uv run scripts/test_vr_input.py
uv run scripts/test_vr_input.py --with-ik    # with IK + Viser visualization

# JoyCon gripper
uv run scripts/test_joycon_gripper.py
```

---

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

Observations flow as typed `Observation` dataclasses in the main process and are converted to plain dicts at the RPC boundary (`obs.to_dict()`).

### Project Layout

```
limb/
  envs/
    launch.py              # Main entry point
    robot_env.py           # dm_env wrapper, builds Observation
    configs/               # YAML → dataclass loader + instantiate()
  agents/
    teleoperation/         # Viser, GELLO, VR agents
    policy_learning/       # pi0, Diffusion Policy agents
  robots/
    yam_motor_chain_robot.py  # CAN bus driver
    inverse_kinematics/       # Pink (Pinocchio) and pyroki (JAX)
  devices/                 # GELLO (Dynamixel), JoyCon gripper, VR client
  visualization/           # URDF viz + camera monitor (Viser)
  sensors/cameras/         # RealSense, OpenCV, ZED drivers
  utils/                   # Rate control, portal RPC, depth utils
configs/                   # YAML launch configs (OmegaConf)
robot_configs/             # Per-arm hardware specs (CAN IDs, PID, joint limits)
scripts/                   # Diagnostics and server scripts
dependencies/              # i2rt submodule, XRoboToolkit SDK, ZED wheel
```

### Config System

Configs are YAML + [OmegaConf](https://github.com/omry/omegaconf). Every component uses `_target_` for dynamic instantiation:

```yaml
agent:
  _target_: limb.agents.teleoperation.yam_viser_agent.YamViserAgent
robots:
  left:
    _target_: limb.robots.yam_motor_chain_robot.YamMotorChainRobot
cameras:
  wrist_left:
    _target_: limb.sensors.cameras.realsense_camera.RealSenseCamera
    serial_number: "XXXXXX"
```

### IK Solvers


| Solver                                         | Library        | Style                   | Notes                               |
| ---------------------------------------------- | -------------- | ----------------------- | ----------------------------------- |
| [Pink](https://github.com/stephane-caron/pink) | Pinocchio + QP | Differential (velocity) | Smooth trajectories, production use |
| [pyroki](https://github.com/chungmin99/pyroki) | JAX            | Global (one-shot)       | Fast startup, less smooth           |


---

## Extending limb

### Add a new agent

```python
from limb.agents.agent import Agent

class MyAgent(Agent):
    def act(self, obs):
        # obs is a plain dict: {"left": {"joint_pos": ...}, "cam_name": {"images": ...}, ...}
        return {"left": {"pos": target_joints}, "right": {"pos": target_joints}}
```

Point your YAML config at it:

```yaml
agent:
  _target_: my_package.MyAgent
  my_param: 42
```

### Add a new robot arm

Implement the `Robot` protocol in `robots/robot.py` — see `yam_motor_chain_robot.py` for a CAN-bus reference implementation.

---

## Development

```bash
uv run ruff check limb/        # lint
uv run ruff check --fix limb/  # auto-fix
uv run ruff format limb/       # format
```

Python 3.11 · `uv` · `ruff` (line length 119) · `loguru` for logging

---

## Acknowledgments

Inspired by [GELLO](https://github.com/wuphilipp/gello_software) and [robots_realtime](https://github.com/uynitsuj/robots_realtime).