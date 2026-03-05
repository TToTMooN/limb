# limb

**Lightweight arm control for robot learning.**

Minimal, ROS-free Python stack for high-frequency control of direct-drive robot arms.
Teleoperation, data collection, and learned policy deployment — everything between bare metal and your research code.

Built for [I2RT YAM](https://github.com/i2rt-robotics/i2rt) bimanual arms.

---

|                          |                                                                              |
| ------------------------ | ---------------------------------------------------------------------------- |
| **No ROS**               | Direct CAN bus at 100 Hz. One process, one config, one launch command.       |
| **Plug-and-play teleop** | Viser web UI, GELLO leader arms, Pico VR — swap with a YAML change.          |
| **Policy-ready**         | Ship your VLA as a server, point limb at it, run inference.                  |
| **Data collection**      | Hands-free episode recording with foot pedal / VR button triggers.           |

**Docs:** [Teleoperation](docs/teleop.md) · [Data Collection](docs/data_collection.md) · [Policy Server Spec](docs/policy_server_spec.md)

---

## Install

```bash
git clone --recurse-submodules https://github.com/TToTMooN/limb
cd limb
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv if needed
uv venv --python 3.11
uv sync
```

> **Submodule:** Includes [i2rt](https://github.com/i2rt-robotics/i2rt) (motor driver) under `dependencies/`. Clone with `--recurse-submodules`.

---

## Hardware Setup

### CAN Interface (YAM arms)

```bash
echo 'SUBSYSTEM=="net", KERNEL=="can*", ACTION=="add", RUN+="/sbin/ip link set %k up type can bitrate 1000000"' \
  | sudo tee /etc/udev/rules.d/99-can.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Verify: `ip link show | grep can` — expect `can_follow_l` and `can_follow_r`.

See [docs/teleop.md](docs/teleop.md) for GELLO and VR hardware setup.

---

## Quick Start

```bash
uv run limb/envs/launch.py --config_path <config.yaml>
```

### Teleoperation

```bash
uv run limb/envs/launch.py --config_path configs/yam_viser_bimanual.yaml    # Viser web UI
uv run limb/envs/launch.py --config_path configs/yam_gello_bimanual.yaml    # GELLO leader arms
uv run limb/envs/launch.py --config_path configs/yam_vr_bimanual.yaml       # Pico VR
```

### Data Collection

Collection configs are overlays — combine with any teleop config:

```bash
uv run limb/envs/launch.py --config_path configs/yam_gello_network_bimanual.yaml configs/collection.yaml
uv run limb/envs/launch.py --config_path configs/yam_vr_bimanual.yaml configs/collection_vr.yaml
```

### Policy Deployment

```bash
uv run limb/envs/launch.py --config_path configs/yam_pi0_bimanual.yaml      # OpenPI (pi0)
uv run limb/envs/launch.py --config_path configs/yam_policy_bimanual.yaml    # Generic WebSocket
```

### Data Tools

```bash
uv run scripts/data/visualize_episode.py --episode_dir recordings/episode_... # Rerun viewer
uv run scripts/data/convert_to_lerobot.py --input_dir recordings/ --output_dir datasets/  # LeRobot format
```

### Diagnostics

```bash
uv run scripts/diagnostics/test_realsense_cameras.py
uv run scripts/diagnostics/test_gello_input.py
uv run scripts/diagnostics/test_vr_input.py
```

---

## Architecture

```
launch.py → control loop @ 100 Hz
  ├─ Agent.act(obs) → action        # teleop device or VLA policy server
  ├─ session.step(obs, action)       # episode recording + trigger management
  ├─ RobotEnv.step(action)           # CAN bus → DM motors
  └─ ViserMonitor.update(obs)        # camera feeds + URDF viz
```

```
limb/
  envs/           # launch.py, robot_env.py, config loader
  agents/         # teleop (Viser, GELLO, VR) + policy (OpenPI, WebSocket)
  robots/         # YAM CAN driver, IK solvers (Pink, pyroki)
  devices/        # GELLO reader, JoyCon gripper, VR client
  recording/      # episode recorder, triggers, session manager
  visualization/  # Viser URDF + camera monitor
  sensors/        # RealSense, OpenCV, ZED camera drivers
configs/          # YAML launch configs
robot_configs/    # per-arm hardware specs
```

---

## Extending limb

```python
from limb.agents.agent import Agent

class MyAgent(Agent):
    def act(self, obs):
        return {"left": {"pos": target_joints}, "right": {"pos": target_joints}}
```

```yaml
agent:
  _target_: my_package.MyAgent
```

---

## Development

```bash
uv run ruff check limb/ && uv run ruff format limb/
```

Python 3.11 · `uv` · `tyro` for CLI · `loguru` for logging · `ruff` (line length 119)

---

Inspired by [GELLO](https://github.com/wuphilipp/gello_software) and [robots_realtime](https://github.com/uynitsuj/robots_realtime).
