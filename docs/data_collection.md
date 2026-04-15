# Data Collection

limb records raw episode data during teleoperation. Post-processing to HDF5, LeRobot, or other formats is done by separate scripts (not part of limb).

---

## Quick Start

Collection configs are overlays — combine with any teleop config:

```bash
# GELLO (network) + keyboard triggers
uv run limb record --config-path configs/yam_gello_network_bimanual.yaml configs/collection.yaml

# GELLO (network) + foot pedal (iKKEGOL double pedal) + keyboard fallback (default)
uv run limb record

# VR + VR button triggers
uv run limb record --config-path configs/yam_vr_bimanual.yaml configs/collection_vr.yaml

# Viser + keyboard triggers
uv run limb record --config-path configs/yam_viser_bimanual.yaml configs/collection.yaml
```

The second YAML merges into the first via OmegaConf, adding the `collection:` block without duplicating teleop/robot/camera config.

See [docs/cli.md](cli.md) for the full CLI reference.

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

**Foot pedal** (`FootPedalTrigger`) — for iKKEGOL / PCsensor double USB pedals:

| Pedal | Signal     | Action                  |
| ----- | ---------- | ----------------------- |
| Left  | START_STOP | Toggle recording on/off |
| Right | DISCARD    | Discard current episode |

Reads via evdev with exclusive grab (key events won't leak to the desktop). Auto-detects the pedal by USB vendor:product ID. Key mapping is configurable via `left_key` / `right_key` in YAML (default: `KEY_A` / `KEY_B`).

**VR buttons** (`VRButtonTrigger`) — for bimanual VR teleop where both hands are occupied:

| Button                | Signal     | Action                  |
| --------------------- | ---------- | ----------------------- |
| B (right controller)  | START_STOP | Toggle recording on/off |
| Y (left controller)   | DISCARD    | Discard current episode |

Note: A/X are already used for arm reset in VR teleop.

**Composite** (`CompositeTrigger`) — combines multiple trigger sources (first signal wins):

```yaml
# Foot pedal + keyboard fallback
trigger:
  _target_: limb.recording.trigger.CompositeTrigger
  sources:
    - _target_: limb.recording.trigger.FootPedalTrigger
    - _target_: limb.recording.trigger.KeyboardTrigger

# VR buttons + keyboard fallback
trigger:
  _target_: limb.recording.trigger.CompositeTrigger
  sources:
    - _target_: limb.recording.trigger.KeyboardTrigger
    - _target_: limb.recording.trigger.VRButtonTrigger
      xr_client: ...  # wire from agent at runtime
```

### Session Workflow

1. Launch with a collection config
2. Press **Space/Enter** (keyboard) or **left pedal** (foot pedal) to start recording
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
    metadata.json                         # config, timing, ee frame names, robot_configs, arm/camera lists
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

`metadata.json` embeds the fully resolved `robot_configs` dict (per arm, with motor chain / CAN channel / PID gains), so `limb replay` can reconstruct the exact hardware the episode was recorded on without any launch config.

Interrupted recordings leave a `RECORDING_IN_PROGRESS` marker file (containing the owning PID). On the next `limb record` startup the incomplete episode is auto-cleaned — provided the owning process is no longer alive, to avoid racing with a concurrent recording.

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

---

## Data Tools

### Replay on Hardware

Before converting, you can verify a recording by streaming its joint commands back to the robot:

```bash
uv run limb replay --episode-dir recordings/red_cube_task/episode_20260304_153045_0001
uv run limb replay --episode-dir recordings/red_cube_task/episode_20260304_153045_0001 --speed 0.5
```

No `--config-path` is needed — the robot config is read from the episode's `metadata.json`.

### Visualize Episodes (Rerun)

View recorded episodes with synchronized joint trajectories, gripper state, EE pose, and camera video:

```bash
uv run limb visualize --episode-dir recordings/red_cube_task/episode_20260304_153045_0001
```

Opens the [Rerun](https://rerun.io) viewer with a timeline scrubber. Per-joint position/velocity traces, gripper state, and camera frames are all time-aligned.

### Convert to LeRobot Format

Convert a session (directory of episodes) to [LeRobot v2.1](https://github.com/huggingface/lerobot) dataset format. No `lerobot` dependency required — only uses pyarrow:

```bash
# Convert all episodes in a session
uv run limb convert-lerobot --input-dir recordings/red_cube_task --output-dir datasets/red_cube

# Only include successful episodes
uv run limb convert-lerobot --input-dir recordings/red_cube_task --output-dir datasets/red_cube --success-only

# Override task instruction
uv run limb convert-lerobot \
  --input-dir recordings/red_cube_task \
  --output-dir datasets/red_cube \
  --task "pick up the red cube and place it in the bowl"

# Push to HuggingFace Hub after conversion
uv run limb convert-lerobot \
  --input-dir recordings/red_cube_task \
  --output-dir datasets/red_cube \
  --push-to-hub myuser/red_cube
```

Incomplete or interrupted episodes (no `metadata.json`, or still carrying a `RECORDING_IN_PROGRESS` marker) are automatically skipped with a warning.

> **Success markers.** `--success-only` filters to episodes that have a `SUCCESS` marker file. The foot-pedal workflow doesn't mark success inline (left pedal is a neutral toggle; right pedal discards), so you'll typically collect first and mark later:
>
> ```bash
> # Mark all as success (if you trust every take)
> uv run limb mark --session-dir recordings/task --all
>
> # Or review interactively and mark one by one
> uv run limb mark --session-dir recordings/task
> ```

### Convert to WebDataset

Convert a session into WebDataset `.tar` shards for streaming training:

```bash
uv run limb convert-webdataset \
  --input-dir recordings/red_cube_task \
  --output-dir datasets/red_cube_wds \
  --samples-per-shard 1000 \
  --jpeg-quality 90
```

### Upload to Cloud Storage

```bash
uv run limb upload --source datasets/red_cube --target s3://my-bucket/datasets/red_cube
uv run limb upload --source datasets/red_cube --target gs://my-bucket/datasets/red_cube
uv run limb upload --source datasets/red_cube --target hf://myuser/red_cube
```

A default target can be configured in `~/.config/limb/storage.yaml` so repeat uploads don't need `--target`.

Output structure:

```
datasets/red_cube/
  meta/
    info.json          # dataset metadata, feature shapes, fps
    stats.json         # per-feature min/max/mean/std
    episodes.jsonl     # episode lengths and task labels
    tasks.jsonl        # task index → instruction string
  data/
    chunk-000/
      episode_000000.parquet   # state + action vectors per frame
      episode_000001.parquet
  videos/
    observation.images.left_wrist_camera/
      chunk-000/
        episode_000000.mp4
    observation.images.right_wrist_camera/
      chunk-000/
        episode_000000.mp4
```
