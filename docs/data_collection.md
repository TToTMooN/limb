# Data Collection

limb records raw episode data during teleoperation. Post-processing to HDF5, LeRobot, or other formats is done by separate scripts (not part of limb).

---

## Quick Start

```bash
# GELLO teleop + keyboard/foot pedal triggers
uv run limb/envs/launch.py --config_path configs/yam_gello_collect.yaml

# VR teleop + VR button triggers
uv run limb/envs/launch.py --config_path configs/yam_vr_collect.yaml
```

---

## Data Collection Sessions

`DataCollectionSession` manages multi-episode collection with hands-free triggers. Add a `collection:` block to any teleop config:

```yaml
collection:
  _target_: limb.recording.session.DataCollectionSession
  num_episodes: 10
  task_instruction: "pick up the red cube and place it in the bowl"
  countdown_s: 3.0
  recorder:
    _target_: limb.recording.episode_recorder.EpisodeRecorder
    base_dir: "recordings/red_cube_task"
    recording_fps: 30
    ee_frame_names:
      left: "ee_link"
      right: "ee_link"
  trigger:
    _target_: limb.recording.trigger.KeyboardTrigger
```

### Trigger Controls

**Keyboard / foot pedal** (`KeyboardTrigger`):

| Key            | Signal     | Action                          |
| -------------- | ---------- | ------------------------------- |
| Space / Enter  | START_STOP | Toggle recording on/off         |
| S              | SUCCESS    | Mark episode as success and save |
| D              | DISCARD    | Discard current episode (deletes data) |
| Q / Escape     | QUIT       | End collection session          |

USB foot pedals that present as keyboard HID work out of the box — most send Enter or Space by default.

**VR buttons** (`VRButtonTrigger`) — for bimanual VR teleop where both hands are occupied:

| Button                | Signal     | Action                  |
| --------------------- | ---------- | ----------------------- |
| B (right controller)  | START_STOP | Toggle recording on/off |
| Y (left controller)   | DISCARD    | Discard current episode |

Note: A/X are already used for arm reset in VR teleop.

**Composite** (`CompositeTrigger`) — combines multiple trigger sources (first signal wins):

```yaml
trigger:
  _target_: limb.recording.trigger.CompositeTrigger
  sources:
    - _target_: limb.recording.trigger.KeyboardTrigger
    - _target_: limb.recording.trigger.VRButtonTrigger
      xr_client: ...  # wire from agent at runtime
```

### Session Workflow

1. Launch with a collection config
2. Press **Space/Enter** (or foot pedal) to start recording
3. Countdown (configurable, default 3s) then recording begins
4. Perform the task via teleop
5. Press **S** to mark success, **Space** to stop (neutral), or **D** to discard
6. Repeat until target episodes reached, or press **Q** to end early
7. Session summary saved as `session_summary.json`

---

## Episode Recording Format

Each episode is saved as a directory:

```
recordings/red_cube_task/
  session_summary.json                    # session-level stats
  episode_20260304_153045_0001/
    metadata.json                         # config, timing, ee frame names, arm/camera lists
    timestamps.npy                        # (N,) float64 Unix timestamps at control rate
    left_states.npz                       # joint_pos (N,6), joint_vel (N,6), gripper_pos (N,1), ee_pose (N,7)
    right_states.npz                      # same structure
    left_actions.npz                      # pos (N,7), optionally vel (N,7)
    right_actions.npz                     # same structure
    left_wrist_camera.mp4                 # video at recording_fps
    left_wrist_camera_timestamps.npy      # (N,) per-frame camera timestamps
    right_wrist_camera.mp4
    right_wrist_camera_timestamps.npy
    head_camera.mp4
    head_camera_timestamps.npy
    SUCCESS                               # marker file (only if marked success)
  episode_20260304_153120_0002/
    ...
```

### Standalone Recording (no session)

For simple always-on recording without episode management, use `recording:` instead of `collection:`:

```yaml
recording:
  _target_: limb.recording.episode_recorder.EpisodeRecorder
  base_dir: "recordings"
  recording_fps: 30
  auto_start: true
  ee_frame_names: {left: "ee_link", right: "ee_link"}
```

---

## Process/Thread Model

Data collection adds no new processes or threads. Everything runs in the main control loop:

```
Process 1 (Portal): Camera 1          ─┐
Process 2 (Portal): Camera 2           │ separate processes for
Process 3 (Portal): Left arm (CAN)     │ hardware I/O isolation
Process 4 (Portal): Right arm (CAN)    │
Process 5 (Portal): Agent (teleop)    ─┘
────────────────────────────────────────
Main process, main thread:
  └─ control loop @ 100 Hz
       ├─ agent.act(obs)               # Portal RPC to agent process
       ├─ session.step(obs, action)    # trigger poll (~0ms) + recording (~1-3ms)
       ├─ env.step(action)             # Portal RPC to robot processes
       └─ monitor.update(obs)          # in-process (viser has own server thread)
────────────────────────────────────────
Episode save (stop_episode): synchronous between episodes (~100-200ms).
Robot holds last commanded position during save.
```

- **TriggerSource.get_signal()** — non-blocking poll, ~0ms (select with 0 timeout for keyboard, bool check for VR)
- **EpisodeRecorder.record()** — list append + cv2.VideoWriter.write, ~1-3ms per step
- **EpisodeRecorder.stop_episode()** — numpy save + video flush, ~100-200ms (runs between episodes)
