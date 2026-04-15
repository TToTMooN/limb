# limb CLI Reference

Single entry point for all common operations. Installed as the `limb` command when you `uv sync`.

```bash
uv run limb <subcommand> [options]
uv run limb --help                   # list all subcommands
uv run limb <subcommand> --help      # flags for one subcommand
```

| Subcommand           | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| `teleop`             | Launch teleoperation (no recording)                    |
| `record`             | Launch a data collection session (teleop + recording)  |
| `devices`            | Discover connected cameras, arms, and input devices    |
| `replay`             | Replay a recorded episode on hardware                  |
| `mark`               | Post-hoc mark recorded episodes as success             |
| `convert-lerobot`    | Convert raw recordings to LeRobot v2.1 dataset format  |
| `convert-webdataset` | Convert raw recordings to WebDataset `.tar` shards     |
| `visualize`          | Open a recorded episode in Rerun                       |
| `upload`             | Push a dataset to S3 / GCS / HuggingFace Hub           |

---

## `limb teleop`

Launch teleoperation from a YAML config. No recording, no data saved.

```bash
uv run limb teleop --config-path configs/yam_viser_bimanual.yaml
uv run limb teleop --config-path configs/yam_gello_network_bimanual.yaml
uv run limb teleop --config-path configs/yam_vr_bimanual.yaml
```

| Flag            | Type        | Default                              | Description                                  |
| --------------- | ----------- | ------------------------------------ | -------------------------------------------- |
| `--config-path` | `list[str]` | `configs/yam_viser_bimanual.yaml`    | One or more YAML files (later overrides earlier) |
| `--log-level`   | `str`       | `INFO`                               | `DEBUG` / `INFO` / `WARNING` / `ERROR`       |

Multiple config paths are merged via OmegaConf — useful for combining a base teleop config with overlays.

See [teleop.md](teleop.md) for hardware setup and backend details.

---

## `limb record`

Launch a data collection session: teleop + hands-free episode recording with a trigger device.

```bash
# GELLO network + foot pedal (default)
uv run limb record

# Explicit config stack
uv run limb record --config-path configs/yam_gello_network_bimanual.yaml configs/collection_pedal.yaml

# VR teleop + VR-button triggers
uv run limb record --config-path configs/yam_vr_bimanual.yaml configs/collection_vr.yaml
```

| Flag            | Type        | Default                                                                              |
| --------------- | ----------- | ------------------------------------------------------------------------------------ |
| `--config-path` | `list[str]` | `(configs/yam_gello_network_bimanual.yaml, configs/collection_pedal.yaml)`           |
| `--log-level`   | `str`       | `INFO`                                                                               |

Collection overlays (`configs/collection*.yaml`) are merged on top of any teleop config. See [data_collection.md](data_collection.md) for trigger setup, metadata, episode format, and session workflow.

Each saved episode embeds the fully resolved robot config into its `metadata.json` so `limb replay` can reconstruct the exact hardware without any launch config.

---

## `limb devices`

Enumerate connected cameras, robot arms (CAN), and input devices. Read-only — nothing is opened for streaming.

```bash
uv run limb devices
uv run limb devices --verbose
```

| Flag          | Type   | Default | Description               |
| ------------- | ------ | ------- | ------------------------- |
| `--verbose`   | `bool` | `False` | Show extra per-device info |

Scans for:
- **RealSense** cameras (serial + model)
- **ZED** cameras (when `pyzed` is installed)
- **CAN** interfaces matching `can*` (state up/down)
- **Dynamixel** serial ports (GELLO USB leaders)
- **Input devices** via evdev: foot pedals, SpaceMouse, etc.

Useful as a smoke test before launching teleop. Generic OpenCV video devices are not enumerated — plug your webcam into limb directly by serial or `/dev/videoN`.

---

## `limb replay`

Stream recorded joint commands back to hardware at configurable speed. Useful for verifying a recording before conversion.

```bash
# Robot config is read from the episode's metadata.json — no --config-path needed
uv run limb replay --episode-dir recordings/task/episode_20260414_200600_0002

# Slow replay at half speed
uv run limb replay --episode-dir recordings/task/episode_20260414_200600_0002 --speed 0.5

# Fallback for legacy recordings that don't embed robot_configs
uv run limb replay --episode-dir <old_episode> --config-path configs/yam_gello_network_bimanual.yaml
```

| Flag             | Type        | Default        | Description                                                    |
| ---------------- | ----------- | -------------- | -------------------------------------------------------------- |
| `--episode-dir`  | `str`       | _required_     | Path to the episode directory (contains `metadata.json` + `*.npz`) |
| `--config-path`  | `list[str]` | `()`           | Fallback robot config for legacy episodes without embedded `robot_configs` |
| `--speed`        | `float`     | `1.0`          | Playback speed multiplier                                      |
| `--log-level`    | `str`       | `INFO`         |                                                                |

Replay uses `actions.npz` when available (includes gripper), falling back to `states.npz` (joint_pos only). Before streaming, it moves the arms to the first pose over 3 seconds.

> Episodes recorded from this branch onward embed the full robot config at save time, so `--config-path` is only needed for older recordings.

---

## `limb mark`

Post-hoc marking of recorded episodes as "success" so they can be filtered at conversion time with `--success-only`. Useful when your collection trigger doesn't mark success inline (e.g. foot pedal) — record everything, review later, commit the good ones.

```bash
# Interactive: step through every episode, answer y/n/s/q
uv run limb mark --session-dir recordings/task

# Batch: mark all episodes in a session as successful
uv run limb mark --session-dir recordings/task --all

# Batch: clear all success markers in a session
uv run limb mark --session-dir recordings/task --clear

# Single episode
uv run limb mark --episode-dir recordings/task/episode_20260414_200600_0002
uv run limb mark --episode-dir recordings/task/episode_20260414_200600_0002 --clear
```

| Flag             | Type   | Default    | Description                                                   |
| ---------------- | ------ | ---------- | ------------------------------------------------------------- |
| `--session-dir`  | `str?` | `None`     | Directory of `episode_*` subdirs (mutually exclusive with `--episode-dir`) |
| `--episode-dir`  | `str?` | `None`     | Path to one episode directory                                 |
| `--all`          | `bool` | `False`    | Session mode only — mark every episode as success             |
| `--clear`        | `bool` | `False`    | Remove SUCCESS markers instead of adding them                 |

**Interactive controls** (default when `--all` / `--clear` are not set):

| Key | Action |
| --- | ------ |
| `y` | Mark success |
| `n` | Unmark |
| `s` / Enter | Skip (no change) |
| `q` | Quit early |

Marking just creates/removes a `SUCCESS` marker file in the episode directory — a trivial, reversible operation.

---

## `limb convert-lerobot`

Convert a session of raw recordings into LeRobot v2.1 dataset format. No `lerobot` package dependency — only uses `pyarrow`.

```bash
uv run limb convert-lerobot \
  --input-dir recordings/pick_up_cube \
  --output-dir datasets/pick_up_cube \
  --task "pick up the grey cube and hand it over" \
  --fps 30 \
  --success-only
```

| Flag              | Type       | Default    | Description                                                  |
| ----------------- | ---------- | ---------- | ------------------------------------------------------------ |
| `--input-dir`     | `str`      | _required_ | Directory of `episode_*` subdirs                             |
| `--output-dir`    | `str`      | _required_ | LeRobot dataset output directory                             |
| `--task`          | `str?`     | `None`     | Task instruction string (defaults to per-episode metadata)   |
| `--robot-type`    | `str`      | `yam`      | Written into `meta/info.json`                                |
| `--fps`           | `int`      | `30`       | Target dataset FPS                                           |
| `--success-only`  | `bool`     | `False`    | Skip episodes without the `SUCCESS` marker                   |
| `--push-to-hub`   | `str?`     | `None`     | `username/repo` to push the dataset to HuggingFace Hub       |

Incomplete episodes (no `metadata.json`, or with `RECORDING_IN_PROGRESS`) are automatically skipped with a warning.

See [data_collection.md](data_collection.md) for the output layout.

---

## `limb convert-webdataset`

Convert recordings into WebDataset `.tar` shards for streaming training pipelines.

```bash
uv run limb convert-webdataset \
  --input-dir recordings/pick_up_cube \
  --output-dir datasets/pick_up_cube_wds \
  --samples-per-shard 1000 \
  --jpeg-quality 90 \
  --fps 30
```

| Flag                   | Type   | Default    | Description                                         |
| ---------------------- | ------ | ---------- | --------------------------------------------------- |
| `--input-dir`          | `str`  | _required_ |                                                     |
| `--output-dir`         | `str`  | _required_ |                                                     |
| `--task`               | `str?` | `None`     | Task instruction override                           |
| `--samples-per-shard`  | `int`  | `1000`     | Frames per `.tar` shard                             |
| `--image-size`         | `int?` | `None`     | Resize images (square) before encoding               |
| `--jpeg-quality`       | `int`  | `90`       | JPEG quality (1–100)                                |
| `--fps`                | `int`  | `30`       |                                                     |
| `--success-only`       | `bool` | `False`    |                                                     |
| `--camera`             | `str?` | `None`     | Include only this camera (default: all cameras)     |

---

## `limb visualize`

Open a recorded episode in [Rerun](https://rerun.io) with joint trajectories, gripper state, EE pose, and camera video on a synchronized timeline.

```bash
uv run limb visualize --episode-dir recordings/task/episode_20260414_200600_0002
```

| Flag            | Type  | Default    |
| --------------- | ----- | ---------- |
| `--episode-dir` | `str` | _required_ |

Uses Rerun (`rerun-sdk`), which is a main dependency — no extras install needed.

---

## `limb upload`

Upload a dataset to cloud storage. Target URI scheme determines the backend:

```bash
# S3 (AWS credential chain)
uv run limb upload --source datasets/pick_up_cube --target s3://my-bucket/datasets/pick_up_cube

# Google Cloud Storage (gcloud credentials)
uv run limb upload --source datasets/pick_up_cube --target gs://my-bucket/datasets/pick_up_cube

# HuggingFace Hub (HF_TOKEN or huggingface-cli login)
uv run limb upload --source datasets/pick_up_cube --target hf://myuser/pick_up_cube
```

| Flag         | Type   | Default    | Description                                                |
| ------------ | ------ | ---------- | ---------------------------------------------------------- |
| `--source`   | `str`  | _required_ | Local dataset directory                                    |
| `--target`   | `str?` | `None`     | Destination URI (falls back to `~/.config/limb/storage.yaml`) |
| `--task`     | `str?` | `None`     | Optional task label for the upload                          |

Default target can be configured in `~/.config/limb/storage.yaml` so repeat uploads don't need `--target`.

---

## Config file conventions

All commands that take `--config-path` accept one or more YAML files. Later files override earlier ones via OmegaConf merge. This is how "overlay" configs work — e.g. adding `configs/collection_pedal.yaml` on top of a teleop config to turn on recording without duplicating robot/camera blocks.

```bash
uv run limb record \
  --config-path configs/yam_gello_network_bimanual.yaml configs/collection_pedal.yaml
```

`_target_` dicts are resolved dynamically via `limb/envs/configs/instantiate.py`, so any dataclass in the codebase can be swapped in from YAML.
