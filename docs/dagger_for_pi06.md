# DAgger data collection + convert for pi0.6 (PiStar)

Practical, command-driven quickstart for collecting DAgger rollouts on YAM
and converting them into the **pistar / pi0.6 RECAP** schema that
[ybpy/pistar](https://github.com/ybpy/pistar) ingests directly.

For the deeper context — the schema, the formulas, why DAgger phase maps onto
pistar `intervention`, and how this fits the broader RECAP loop — read
[`docs/recap_collection.md`](recap_collection.md). For the training side, read
[`openpi/docs/recap_finetune.md`](../openpi/docs/recap_finetune.md). This doc
is the minimal happy path.

---

## Working reference dataset

A concrete output you can inspect lives at:

```
datasets/vial_rollout_v1/   # 10 DAgger episodes, 3 SUCCESS / 7 FAILURE, 21286 frames
```

It was produced by exactly the commands below from the recording at
`recordings/vial_placement_intervention_demonstrations_20260522_192006`.

---

## Prerequisites

1. **A pi0.5 SFT checkpoint served on `0.0.0.0:8111`.** See
   [`openpi/docs/yam_finetune.md`](../openpi/docs/yam_finetune.md). RECAP
   reshapes a working policy; it can't fix one that produces gibberish.
2. The DAgger pedal trigger configured on `/dev/input/event9`
   (iKKEGOL/PCsensor — auto-detected).
3. Keyboard available (the dagger collection session uses `s`/SPACE/`d`/`q`
   for episode lifecycle).

---

## Step 1 — Collect

```bash
uv run limb record \
  --config-path configs/yam_dagger_pi0_bimanual.yaml \
                configs/dagger_collection.yaml
```

The agent boots in `initial_phase: paused` (robot holds), session waits for
SPACE. Per episode:

1. **(Optional)** right pedal → CORRECTING, back-drive followers to the start
   pose, right pedal → PAUSED.
2. Stage the scene.
3. **SPACE** → recording starts.
4. **Left pedal** → AUTONOMOUS; policy runs. Intervene with **left pedal** →
   PAUSED → **right pedal** → CORRECTING when needed, then **right pedal** →
   PAUSED → **left pedal** → AUTONOMOUS to hand back.
5. End the episode:
   - **`s`** → SUCCESS
   - **SPACE** → FAILURE (the policy didn't finish the task)
   - **`d`** → discard
   - **`q`** → quit the session

Each episode produces `recordings/<task>_<ts>/episode_<...>/` with:

```
left_actions.npz       right_actions.npz       (follower commanded actions)
left_states.npz        right_states.npz
left_policy_actions.npz right_policy_actions.npz
leader_left_actions.npz leader_right_actions.npz        (operator side)
leader_left_states.npz  leader_right_states.npz
head_camera.mp4 left_wrist_camera.mp4 right_wrist_camera.mp4 (+ *_timestamps.npy)
phase.npy              (autonomous / paused / correcting per tick)
interventions.npy      (bool per tick)
correction_index.npy   (per-correction id)
timestamps.npy
SUCCESS  or  FAILURE   (marker file from your s/SPACE label)
metadata.json
```

Aim for **20–50 episodes of mixed outcomes** — RECAP needs both classes to
train the value model. Don't filter at collection time.

---

## Step 2 — Convert for pi0.6 (the headline command)

```bash
uv run limb convert-lerobot \
  --input-dir recordings/<your_session_dir> \
  --output-dir datasets/<task>_pistar_v1 \
  --target-fps 30 \
  --include-arms left right \
  --pistar \
  --task "<your task instruction>"
```

The three flags that matter for pi0.6:

| Flag | What it does |
|---|---|
| `--pistar` | Emit the five pistar/pi0.6 RECAP columns into each episode's data parquet: `intervention`, `reward`, `reward_label`, `value_label`, `adv_ind`. Formulas mirror `pistar_rlds_demo_processing.py`. |
| `--include-arms left right` | DAgger records **four** arms (followers + leaders). Without this filter the converted state/action is 28-D, not the 14-D pi0.5/pi0.6 expects. |
| `--target-fps 30` | Standard pi0.5/pi0.6 dataset rate. The DAgger control loop runs ~28.5 Hz; the converter resamples to a regular 30 Hz grid. |

The exact command used to produce the reference dataset
(`datasets/vial_rollout_v1`):

```bash
uv run limb convert-lerobot \
  --input-dir recordings/vial_placement_intervention_demonstrations_20260522_192006 \
  --output-dir datasets/vial_rollout_v1 \
  --target-fps 30 \
  --include-arms left right \
  --pistar \
  --task "Use one arm to grasp the papercup and hand it over to the other arm"
```

### What `--pistar` derives, per frame

| Column | dtype | Derived as |
|---|---|---|
| `intervention` | int64 | `1` if `phase == "correcting"`, else `0` |
| `reward` | float32 | `1.0` at the last frame iff SUCCESS marker; `0.0` everywhere else |
| `reward_label` | float32 | `-1/T` per step, `0.0` at terminal (same regardless of outcome) |
| `value_label` | float32 | success → linear ramp `-(T-1-t)/T` (≈ −1 → 0); failure → constant `-1.0` |
| `adv_ind` | string | `"positive"` on `intervention=1` frames; `"none"` on `intervention=0` (the VLM will rewrite these to `"positive"` / `"negative"` in pistar Stage 5) |

All standard LeRobot v3.0 columns (`observation.state`, `observation.images.*`,
`action`, `task_index`, etc.) are also written normally.

### SFT bootstrap variant — `--pistar-demo`

For the **pi0.5 SFT bootstrap** dataset (collected with gello teleop, not
DAgger — operator drives every frame, no `phase=="correcting"` signal):

```bash
uv run limb convert-lerobot \
  --input-dir recordings/<gello_sft_session> \
  --output-dir datasets/<task>_demo \
  --target-fps 30 \
  --include-arms left right \
  --pistar-demo
```

`--pistar-demo` forces `intervention=1`, treats every episode as success, and
sets `adv_ind="positive"` for every frame — exactly matching pistar's
`pistar_rlds_demo_processing.py` demo conversion. Use this for the initial
SFT dataset that bootstraps the pi0.6 checkpoint.

---

## Step 3 — Verify

Spot-check the converted dataset before training:

```bash
uv run python -c "
import json, pyarrow.parquet as pq
from pathlib import Path
out = Path('datasets/vial_rollout_v1')          # or your path
info = json.loads((out / 'meta/info.json').read_text())
print(f'Episodes: {info[\"total_episodes\"]}  Frames: {info[\"total_frames\"]}  FPS: {info[\"fps\"]}')
for c in ['intervention','reward','reward_label','value_label','adv_ind']:
    assert c in info['features'], f'missing {c}'
    print(f'  feature {c:14s} dtype={info[\"features\"][c][\"dtype\"]}')
# Per-episode check
n_success, n_intv, n_total = 0, 0, 0
for pq_file in sorted((out / 'data').rglob('*.parquet')):
    t = pq.read_table(pq_file).to_pandas()
    n_total += len(t)
    n_intv  += int(t['intervention'].sum())
    if t['reward'].sum() == 1.0:
        n_success += 1
print(f'Successful episodes: {n_success}')
print(f'Intervention rate:   {100*n_intv/n_total:.1f}%')
"
```

Healthy output for the reference dataset:

```
Episodes: 10  Frames: 21286  FPS: 30
  feature intervention   dtype=int64
  feature reward         dtype=float32
  feature reward_label   dtype=float32
  feature value_label    dtype=float32
  feature adv_ind        dtype=string
Successful episodes: 3
Intervention rate:   32.7%
```

Sanity bands for a typical DAgger run:

- **Intervention rate** between 10% and 60%. Too low → operator wasn't
  catching failures; too high → policy is too far gone, SFT a stronger
  starting checkpoint first.
- **Success rate** between 20% and 70%. Pure-success or pure-failure data
  collapses the value distribution — need both classes.

---

## Step 4 — Hand off to pistar (v3 → v2.1 + train)

PiStar / openpi read **LeRobot v2.1**; limb emits v3.0. Convert once:

```bash
uv run python openpi/scripts/convert_v3_to_v21.py \
  --src=datasets/vial_rollout_v1 \
  --dst=datasets/vial_rollout_v1_v21
```

The script **symlinks** the v3.0 data parquets under v2.1 names
(`file-NNN.parquet` → `episode_NNNNNN.parquet`) and only rewrites the meta
files. The parquet contents — including all five pistar columns and the
`adv_ind` string column — are byte-identical to the v3.0 source, so nothing
can be dropped on this hop.

Then continue with the pistar pipeline:

1. **Stage 1** — initial PiStar checkpoint (`scripts/train.py` with
   `pistar=True` in the model config) on a `--pistar-demo` dataset.
2. **Stage 2–3** — collect rollouts via this DAgger flow, convert with
   `--pistar`, merge demo + rollout with pistar's `scripts/merge_datasets.py`.
3. **Stage 4 + Stage 5** — VLM value model training and `adv_ind` relabel.
   Currently blocked on VLM base weights (AliPan) and a broken pistar
   import (`ValueModelWeightLoader`). **Plan C skips these** — see
   [`docs/pi06_train_runbook.md`](pi06_train_runbook.md).
4. **Stage 6 (Plan C: now)** — `scripts/train.py pi06_yam_vial_30fps`
   directly on the limb-supplied `adv_ind`. Full commands in the runbook
   above.
5. **Serve** — vanilla `serve_policy.py` with `adv_ind_input="positive"` in
   the model config; limb's `OpenPIClient` and prompt are unchanged.

---

## Flag reference (limb convert-lerobot)

| Flag | Default | Notes |
|---|---|---|
| `--pistar` | `False` | DAgger rollout mode. Use this for any DAgger-collected dataset feeding pistar Stage 2+. |
| `--pistar-demo` | `False` | SFT demo mode. Use for gello/teleop demos feeding pistar Stage 1. Mutually exclusive with `--pistar` in spirit; both write the same column set but with different values. |
| `--include-arms` | `None` | Whitelist of arm names. Always set to `left right` for YAM bimanual policy training. |
| `--target-fps` | auto-detect | Set to `30` for the standard pi0.5/pi0.6 dataset rate. |
| `--success-only` | `False` | Drops FAILURE episodes. **Do not use for RECAP** — the value model needs both classes. |
| `--task` | first episode's metadata | Override the dataset-wide task instruction. |

---

## Troubleshooting

**Q. Converter says "state dim: 28" instead of 14.**
A. You forgot `--include-arms left right`. DAgger records all four arms;
filter to the followers.

**Q. `adv_ind` column is all `"none"` after `--pistar`.**
A. Your episodes have no CORRECTING frames — either no DAgger phase trigger
was active or the operator never intervened. For SFT data use
`--pistar-demo` instead.

**Q. `value_label` is all `-1.0` for every episode.**
A. Every episode is FAILURE. Collect some successful runs (or, if everything
really failed, that's a signal the starting policy isn't ready for DAgger —
SFT more first).

**Q. `reward.sum()` over a SUCCESS episode is not exactly 1.0.**
A. Resampling shouldn't change this — `reward` is computed *after* resample
at the converter's terminal frame. If you see this, file a bug.

---

## See also

- [`docs/recap_collection.md`](recap_collection.md) — full schema + the why
- [`openpi/docs/recap_finetune.md`](../openpi/docs/recap_finetune.md) — pistar training pipeline
- [`docs/data_collection.md`](data_collection.md) — generic (non-DAgger) collection
- [`docs/teleop.md`](teleop.md) — teleop modes
- [ybpy/pistar](https://github.com/ybpy/pistar) — the pi0.6 / PiStar implementation
