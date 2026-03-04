# R1 Lite GELLO Teleoperation

Use the Galaxea R1 Lite Teleop as a GELLO (direct joint-to-joint) leader device
to control YAM arms.  The R1 Lite Teleop and the YAM share the same kinematic
structure so joint angles transfer directly — no inverse kinematics needed.

## Overview

The R1 Lite Teleop contains 12 Dynamixel servos (6 per arm, IDs 1–12) read by
an onboard ARM computer (LubanCat board).  Since the newer hardware no longer
exposes a USB serial port externally, joint positions are streamed over Ethernet
via a lightweight TCP server.

```
┌─────────────────────────────────┐         ┌───────────────────────────┐
│  R1 Lite Teleop (10.42.0.1)    │  TCP    │  Host PC                  │
│                                 │ :5555   │                           │
│  Dynamixel servos               │────────▶│  NetworkDynamixelReader   │
│    └─ /dev/r1litet_usb          │  Ether  │    └─ YamGelloAgent       │
│    └─ gello_position_server.py  │   net   │       └─ YAM arms (CAN)  │
└─────────────────────────────────┘         └───────────────────────────┘
```

## Hardware Setup

1. **Fix the R1 Lite Teleop** to a table using the provided G-clamps.
2. **Place all linkages in the zero/home position** before powering on.
3. **Connect Ethernet** from R1 Lite Teleop(R) to the host PC.
4. **Power on** the R1 Lite Teleop via the DC adapter.
5. **Configure the PC's Ethernet** to "Shared to other computers" so the
   teleop device gets IP `10.42.0.1` and the PC gets `10.42.0.x`.

Verify connectivity:

```bash
ping 10.42.0.1
```

## SSH Access

The onboard computer credentials (per Galaxea docs):

- **Username:** `cat`
- **Password:** `temppwd`

Set up passwordless SSH (one-time):

```bash
ssh-keygen -t ed25519          # skip if you already have a key
ssh-copy-id cat@10.42.0.1     # enter password: temppwd
```

Verify:

```bash
ssh cat@10.42.0.1 "uname -a"
```

## Calibration

The server performs **zero-pose calibration** at startup, following the same
approach as Galaxea's own `gello_teleop.py`:

1. **Before starting the server**, place both teleop arms in the **rest/zero
   pose** — all linkages seated in the positioning slots (see Galaxea docs
   section 2.2 for the reference photo).
2. **Start the server** — it reads the raw encoder positions at this pose and
   stores them as offsets.
3. All subsequent readings are `(raw_signed_angle - offset)`, giving
   **calibrated joint angles where 0 = rest pose**.
4. These calibrated angles map 1:1 to YAM joint angles — the agent sends them
   directly as position commands (no delta/offset math on the PC side).

The server auto-detects the board type (LubanCat-4) and applies the correct
joint sign convention automatically.

## Architecture

### What runs on the R1 Lite Teleop

**`gello_position_server.py`** — a single self-contained Python script deployed
to the device.  It:

- Opens `/dev/r1litet_usb` (the Dynamixel servo chain) at 4 Mbps
- **Calibrates** by reading positions at the rest pose and storing as offsets
- Reads all 12 motor positions at ~1 kHz via `GroupSyncRead` (Dynamixel SDK
  Protocol 2.0)
- Subtracts zero-pose offsets so output is calibrated (0 = rest pose)
- Auto-detects LubanCat-4 board and applies correct joint signs
- Disables torque so the arms are free to move (read-only leader)
- Runs a TCP server on port 5555
- Streams calibrated positions to connected clients at 200 Hz

**Dependencies** are all pre-installed on the device:

| Package        | Source                                          |
| -------------- | ----------------------------------------------- |
| Python 3.10    | System                                          |
| numpy 1.21     | System pip                                      |
| dynamixel_sdk  | `/opt/ros/humble/local/lib/python3.10/dist-packages` |

The script auto-adds the ROS 2 Python path to `sys.path` so no `source
/opt/ros/humble/setup.bash` is needed.

**Compatibility with existing Galaxea software:**

- Does NOT modify any Galaxea files, configs, or systemd services
- Does NOT interfere with `node_monitor.service` (torso monitor uses ROS 2
  DDS over Ethernet, not USB serial)
- Cannot run simultaneously with the Galaxea teleop session
  (`robot_startup.sh`) since both would use `/dev/r1litet_usb`

### What runs on the host PC

| Component | File | Purpose |
| --- | --- | --- |
| `NetworkDynamixelReader` | `limb/dynamixel/network_dynamixel_reader.py` | TCP client — connects to the server, receives position frames, exposes `.get_joint_positions()` |
| `DynamixelReader` | `limb/dynamixel/dynamixel_reader.py` | Direct USB serial reader (for older hardware with micro USB) |
| `YamGelloAgent` | `limb/agents/teleoperation/yam_gello_agent.py` | Teleoperation agent — direct 1:1 mapping: `yam_target = clamp(leader_position, joint_limits)` |

The agent auto-selects the reader based on config:

- If `host` is set → `NetworkDynamixelReader` (Ethernet)
- If `host` is not set → `DynamixelReader` (USB serial)

Since positions are calibrated server-side (0 = rest pose), the agent simply
forwards them to the YAM, clamped to joint limits.

### TCP Protocol

1. Client connects to `<teleop_ip>:5555`
2. Server sends a 4-byte header: `uint32` big-endian = number of joints (12)
3. Server then continuously sends frames of `12 × 8 = 96` bytes (`float64`
   big-endian, positions in radians) at 200 Hz
4. Connection closes on client disconnect or Ctrl+C on server

## Quick Start

### 1. Deploy the server (one-time)

```bash
scp scripts/gello_position_server.py cat@10.42.0.1:~/
```

### 2. Place arms in rest pose and start the server

**Important:** Place both teleop arms in the zero/rest pose before starting.

```bash
ssh cat@10.42.0.1 "python3 ~/gello_position_server.py"
```

You should see:

```
Board: Embedfire LubanCat-4 V1 (LubanCat-4: True)
Joint signs: [1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]
Capturing zero-pose offsets (arms must be in rest position)...
Calibrated. Offsets: [0.014, -4.7, 6.245, ...]
Polling 12 motors
Listening on 0.0.0.0:5555 (streaming at 200 Hz)
Waiting for client...
```

After "Calibrated" appears, the arms are free to move.

To run in the background:

```bash
ssh cat@10.42.0.1 "nohup python3 ~/gello_position_server.py > /tmp/gello_server.log 2>&1 &"
```

### 3. Test on the host PC

**Raw joint positions** (no robot hardware needed):

```bash
.venv/bin/python scripts/test_gello_input.py --host 10.42.0.1
```

**Agent loop with mock robot** (no real YAM needed):

```bash
.venv/bin/python scripts/test_gello_input.py --host 10.42.0.1 --with-agent
```

**Full teleop with real YAM arms:**

```bash
uv run limb/envs/launch.py --config_path configs/yam_gello_network_bimanual.yaml
```

### 4. Stop the server

```bash
ssh cat@10.42.0.1 "pkill -f gello_position_server"
```

## Configuration

### Network mode (Ethernet)

`configs/yam_gello_network_bimanual.yaml`:

```yaml
agent:
  _target_: limb.agents.teleoperation.yam_gello_agent.YamGelloAgent
  host: "10.42.0.1"
  network_port: 5555
  bimanual: true
  left_motor_ids: [1, 2, 3, 4, 5, 6]
  right_motor_ids: [7, 8, 9, 10, 11, 12]
  default_gripper_value: 0.0
```

### Direct USB mode (legacy)

`configs/yam_gello_bimanual.yaml`:

```yaml
agent:
  _target_: limb.agents.teleoperation.yam_gello_agent.YamGelloAgent
  port: "/dev/ttyUSB0"
  baudrate: 4000000
  bimanual: true
  left_motor_ids: [1, 2, 3, 4, 5, 6]
  right_motor_ids: [7, 8, 9, 10, 11, 12]
  joint_signs_left: [1, 1, -1, -1, -1, 1]
  joint_signs_right: [1, 1, -1, -1, -1, 1]
  default_gripper_value: 0.0
```

### Server-side options

```
python3 gello_position_server.py --help

  --device         USB serial path       (default: /dev/r1litet_usb)
  --baudrate       Serial baudrate       (default: 4000000)
  --tcp-port       TCP listen port       (default: 5555)
  --hz             Streaming rate in Hz  (default: 200)
  --motor-ids      Motor IDs to read     (default: 1 2 3 ... 12)
  --joint-signs    Per-joint sign (+1/-1)(default: all +1)
```

## File Summary

| File | Location | Description |
| --- | --- | --- |
| `gello_position_server.py` | `scripts/` (and deployed to `~/` on device) | TCP position server for the R1 Lite Teleop |
| `network_dynamixel_reader.py` | `limb/dynamixel/` | TCP client reader (host PC) |
| `dynamixel_reader.py` | `limb/dynamixel/` | Direct USB serial reader (legacy) |
| `yam_gello_agent.py` | `limb/agents/teleoperation/` | GELLO teleoperation agent |
| `test_gello_input.py` | `scripts/` | Diagnostic / test script |
| `yam_gello_network_bimanual.yaml` | `configs/` | Config for Ethernet mode |
| `yam_gello_bimanual.yaml` | `configs/` | Config for USB serial mode |

## Troubleshooting

**"Connection refused" when running test script**
— The server isn't running on the teleop device. SSH in and start it.

**"Cannot open /dev/r1litet_usb" on the server**
— The USB serial device isn't detected. Check that the Dynamixel chain is
properly connected to the LubanCat board. Run `ls /dev/r1litet_usb`.

**Positions don't change when moving the arms**
— Verify the server log shows "Polling 12 motors". If positions are constant,
check that torque is disabled (the server does this automatically) and that the
arms are mechanically free.

**"Cannot run simultaneously with Galaxea teleop"**
— If the Galaxea teleop session (`robot_startup.sh`) is running, it holds
`/dev/r1litet_usb`. Stop it first: `ssh cat@10.42.0.1 "pkill -f robot_startup"`.

## Reference
https://docs.galaxea-ai.com/zh/Guide/R1Lite/isomorphic_teleop/unpacking/R1Lite_Teleopration_unpacking/#531-launch