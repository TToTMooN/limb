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
  policy_server_spec.md   # Spec for companion policy server repo
dependencies/             # Git submodules: i2rt, XRoboToolkit
```

### Control loop

```
launch.py
  └─ RobotEnv.step() @ 100 Hz
       ├─ Agent.act(obs) → action
       │    teleop:  device → IK → joint targets
       │    policy:  obs → PolicyClient.infer() → action chunk → joint targets
       ├─ Robot.command(action) → CAN bus → DM motors
       ├─ ViserMonitor.update(obs)      # optional: camera feeds + URDF
       └─ EpisodeRecorder.record(obs, action)  # optional: raw data capture
```

### Observation format

`Observation.to_dict()` produces:
```python
{
  "timestamp": float,
  "left": {"joint_pos": (6,), "joint_vel": (6,), "gripper_pos": (1,), "ee_pose": (7,)},
  "right": {"joint_pos": (6,), "joint_vel": (6,), "gripper_pos": (1,), "ee_pose": (7,)},
  "left_wrist_camera": {"images": {"rgb": (H,W,3)}, "timestamp": float},
  "right_wrist_camera": {"images": {"rgb": (H,W,3)}, "timestamp": float},
}
```

### Action format

```python
{"left": {"pos": (7,)}, "right": {"pos": (7,)}}  # 6 joints + 1 gripper
# If use_joint_state_as_action: also includes "vel": (7,) per arm
```

---

## Launch Commands

### Teleoperation

```bash
uv run limb/envs/launch.py --config_path configs/yam_viser_bimanual.yaml
uv run limb/envs/launch.py --config_path configs/yam_gello_bimanual.yaml
uv run limb/envs/launch.py --config_path configs/yam_vr_bimanual.yaml
```

### VLA Policy Deployment

```bash
# OpenPI (pi0/pi0-FAST) — requires OpenPI server at host:port
uv run limb/envs/launch.py --config_path configs/yam_pi0_bimanual.yaml

# Generic policy server — requires server implementing docs/policy_server_spec.md
uv run limb/envs/launch.py --config_path configs/yam_policy_bimanual.yaml
```

### Hardware Diagnostics

```bash
uv run scripts/test_realsense_cameras.py
uv run scripts/test_gello_input.py
uv run scripts/test_vr_input.py
```

### GELLO Network Mode (R1 Lite)

```bash
bash scripts/start_gello_server.sh        # Start on R1 Lite (10.42.0.1)
bash scripts/start_gello_server.sh --kill  # Kill remote server
```

---

## VLA Policy Integration

Policies run as external servers. limb connects via `PolicyClient`:

```
┌─ Robot machine (limb) ──────────┐     ┌─ GPU machine ────────────┐
│ RobotEnv → obs                  │     │ Policy server             │
│ YamPolicyAgent                  │     │  (OpenPI / LeRobot / etc) │
│   ├─ ObsTransform → flat obs   │────▶│  model.infer(obs)         │
│   ├─ PolicyClient.infer()      │◀────│  → action chunk           │
│   ├─ ActionChunkManager        │     └───────────────────────────┘
│   └─ ActionTransform → action  │
│ Robot.command(action)           │
└─────────────────────────────────┘
```

**PolicyClient implementations:**
- `OpenPIClient` — wraps `openpi_client` for pi0/pi0-FAST/pi0.5
- `WebSocketPolicyClient` — generic msgpack+WebSocket (see `docs/policy_server_spec.md`)

**Transforms** are YAML-configurable (no hardcoded key mappings):
- `ObsTransform` / `OpenPIObsTransform` — key remapping, image resize, state concatenation
- `ActionTransform` — bimanual split, gripper clip

**Action chunking:** `ActionChunkManager` buffers multi-step action chunks with
temporal smoothing (weighted linear interpolation across overlapping chunks).

---

## Episode Recording

`EpisodeRecorder` captures raw control loop data. Configured in YAML:

```yaml
recording:
  _target_: limb.recording.episode_recorder.EpisodeRecorder
  base_dir: "recordings"
  recording_fps: 30
  auto_start: true
  ee_frame_names: {left: "ee_link", right: "ee_link"}
```

Output per episode:
```
recordings/episode_20260304_153045_0001/
  metadata.json              # config, timing, ee frame names, arm/camera lists
  timestamps.npy             # (N,) float64 Unix timestamps at control rate
  left_states.npz            # joint_pos (N,6), joint_vel (N,6), gripper_pos (N,1), ee_pose (N,7)
  right_states.npz
  left_actions.npz           # pos (N,7)
  right_actions.npz
  left_wrist_camera.mp4      # video
  left_wrist_camera_timestamps.npy  # per-frame camera timestamps
  ...
```

Post-processing to HDF5/LeRobot/other formats is done by separate scripts (not in limb).

---

## Config System

YAML files with `_target_` for dynamic instantiation (Hydra-like, but custom):

```yaml
_target_: limb.envs.launch.LaunchConfig
hz: 100.0
sensors:
  cameras:
    left_wrist_camera:
      _target_: limb.sensors.cameras.camera.CameraNode
      camera:
        _target_: limb.sensors.cameras.realsense_camera.RealsenseCamera
        serial_number: "409122274017"
robots:
  left: ["robot_configs/yam/left.yaml"]
  right: ["robot_configs/yam/left.yaml", "robot_configs/yam/right.yaml"]
agent:
  _target_: limb.agents.teleoperation.yam_viser_agent.YamViserAgent
  bimanual: true
  ik_solver: "pink"
```

Robot configs (`robot_configs/yam/left.yaml`, `right.yaml`) specify motor chain
(CAN IDs, motor types, interface name), PID gains, joint limits, URDF path.

---

## Hardware Setup

### CAN Interface (YAM arms)

```bash
echo 'SUBSYSTEM=="net", KERNEL=="can*", ACTION=="add", RUN+="/sbin/ip link set %k up type can bitrate 1000000"' \
  | sudo tee /etc/udev/rules.d/99-can.rules
sudo udevadm control --reload && sudo udevadm trigger
```

Verify: `ip link show | grep can` — expect `can_follow_l` and `can_follow_r`.

### GELLO (Dynamixel)

USB-to-serial at 4 Mbps. Network mode: TCP at `10.42.0.1:port`.

### VR (Pico headset)

```bash
bash scripts/install_xrobotoolkit_sdk.sh
```

---

## IK Solvers

| Solver | Library | Style | Use When |
|--------|---------|-------|----------|
| `YamPink` | Pinocchio + QP | Differential (velocity) | Production, smooth |
| `YamPyroki` | JAX | Global (one-shot) | Fast startup |

Both support bimanual and expose `self.joints` dict.

---

## Development Conventions

- **Package manager**: `uv` (not pip)
- **Python**: 3.11
- **Linter**: `ruff` (line length 119)
- **Logging**: `loguru` (`from loguru import logger`)
- **Rate control**: `Rate` class from `robots/utils.py`
- **Multi-process RPC**: `portal` library with `@remote()` decorator
- **Config**: OmegaConf + custom `instantiate()` (not Hydra)
- **Lazy imports**: Optional deps (openpi_client, websockets) imported inside methods

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
```

Install: `uv sync`
