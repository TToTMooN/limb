# Teleoperation

Every launch follows the same pattern — one entrypoint, one YAML config:

```bash
uv run limb/envs/launch.py --config_path <config.yaml>
```

---

## Viser (web browser)

3D scene with URDF visualization, camera feeds, and interactive Cartesian handles.
IK via [Pink](https://github.com/stephane-caron/pink) (Pinocchio QP) or [pyroki](https://github.com/chungmin99/pyroki) (JAX).

```bash
# Bimanual (open browser at localhost:8080)
uv run limb/envs/launch.py --config_path configs/yam_viser_bimanual.yaml

# Single arm
uv run limb/envs/launch.py --config_path configs/yam_viser_single_arm.yaml
```

---

## GELLO (Dynamixel leader arms)

Direct joint-to-joint mapping — leader and follower share kinematics, no IK needed.

> **Note:** USB mode is not yet fully verified. Network mode (R1 Lite) is the recommended setup.

```bash
# USB (direct connection)
uv run limb/envs/launch.py --config_path configs/yam_gello_bimanual.yaml
```

### GELLO Network Mode (R1 Lite)

Start the position server on the remote device, then launch locally:

```bash
# On R1 Lite
bash scripts/start_gello_server.sh              # default 10.42.0.1
bash scripts/start_gello_server.sh 10.42.0.2    # custom host
bash scripts/start_gello_server.sh --kill        # stop server

# On workstation
uv run limb/envs/launch.py --config_path configs/yam_gello_network_bimanual.yaml
```

---

## VR (Pico headset)

Cartesian control via Pico VR controllers through [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service).

### Setup

1. Install the [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) on your workstation.
2. Install the Python SDK:
   ```bash
   bash scripts/install_xrobotoolkit_sdk.sh
   ```
3. Open the **XRoboToolkit** app on the Pico headset and confirm poses stream in the PC Service.

### Launch

```bash
uv run limb/envs/launch.py --config_path configs/yam_vr_bimanual.yaml
```

### Button Mapping

| Button             | Function                                           |
| ------------------ | -------------------------------------------------- |
| **Grip** (hold)    | Activate arm — arm follows hand                    |
| **Grip** (release) | Freeze arm — re-grip to continue from new position |
| **Trigger**        | Gripper (0 = open, fully pressed = closed)         |
| **A / X**          | Reset arm to home position                         |

The Viser web UI stays active during VR teleop for monitoring.

---

## IK Solvers

| Solver                                         | Library        | Style                   | Notes                               |
| ---------------------------------------------- | -------------- | ----------------------- | ----------------------------------- |
| [Pink](https://github.com/stephane-caron/pink) | Pinocchio + QP | Differential (velocity) | Smooth trajectories, production use |
| [pyroki](https://github.com/chungmin99/pyroki) | JAX            | Global (one-shot)       | Fast startup, less smooth           |

Both support bimanual and expose the same interface. Select via `ik_solver: "pink"` or `ik_solver: "pyroki"` in the agent config.
