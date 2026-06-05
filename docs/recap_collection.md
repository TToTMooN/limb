# RECAP data collection on YAM bimanual

Robot-side recipe for collecting **RECAP** rollout data with limb. The
training side lives in openpi (`openpi/docs/recap_finetune.md`); this doc
covers what limb has to record and convert.

> **TL;DR.** We follow **pistar** ([ybpy/pistar](https://github.com/ybpy/pistar))
> — the **direct pi0.6/RECAP implementation** that the algorithm originates
> from. pistar is a fork of openpi, conditions on advantage via a tokenized
> `adv_ind ∈ {"positive","negative","none"}` consumed by openpi's standard
> tokenizer transform, and ships a real-robot deployment toolkit
> (`control_your_robot/`). For limb, the **recording side is already done**;
> the converter needs to write **five pistar fields** per frame:
> `intervention`, `reward`, `reward_label`, `value_label`, `adv_ind`. All
> derivable from data already recorded — see
> [§ What's left to do](#whats-left-to-do).

> **What RECAP is** — *RL with Experience and Corrections via
> Advantage-conditioned Policies*. The offline RL algorithm in **pi0.6**.
> Stages (offline): (1) merge demo + rollout data, (2) train a VLM value
> model on per-step `value_label`, (3) run VLM inference to rewrite `adv_ind`
> on rollout frames, (4) continue policy training with advantage-conditioned
> tokens. Limb owns Stage 0 (data).
>
> **References:**
>
> - **Algorithm we use: [ybpy/pistar](https://github.com/ybpy/pistar)** — pi0.6
>   itself. Fork of openpi (JAX). Conditioning is a tokenized `adv_ind`
>   string (`"positive"`, `"negative"`, `"none"`) consumed in openpi's
>   standard tokenizer (`src/openpi/transforms.py:276`). Real-robot
>   deployment via `control_your_robot/`. Value model = SigLIP + Gemma3 +
>   201-atom C51 head over [-1, 0].
> - **Alternative: [RLinf `examples/recap/`](https://github.com/RLinf/RLinf)**
>   ([docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)).
>   Same value-model architecture and tokenized-advantage conditioning, but
>   **PyTorch** and a *re-implementation* of pi0.6's RECAP. Labels advantage
>   via Critic-Expert + N-step lookahead → top-30% quantile binarization,
>   instead of pistar's VLM labeling. Sim-validated only (LIBERO). The
>   vendored standalone `openpi/scripts/recap/compute_returns.py` belongs to
>   this alternative; not used in the pistar path.
> - **Reference: [MINT-SJTU/Evo-RL](https://github.com/MINT-SJTU/Evo-RL).**
>   Real-robot RECAP but with a **different conditioning** (advantage tag
>   injected into task text). Useful only for collection-protocol intuition
>   (`s`/`f` per-episode labels, penalty scaling).

---

## What RECAP needs from rollout data (pistar schema)

pistar requires five RECAP fields per frame, **on top of** the standard
LeRobot columns. From `pistar_rlds_demo_processing.py` and the pistar README:

| Column | Dtype / shape | Meaning | Formula at convert time |
|---|---|---|---|
| `intervention` | int64 [1] | `1` if human-driven this frame, `0` if autonomous | `phase == "correcting"` |
| `reward` | float32 [1] | sparse success signal | `1.0` at last frame iff episode SUCCESS, else `0.0` |
| `reward_label` | float32 [1] | per-step reward used by VLM advantage | `-1/T` except `0.0` at terminal (same regardless of outcome) |
| `value_label` | float32 [1] | training target for VLM value model | success → `-(T-1-t)/T` (linear ramp `≈-1 → 0`); failure → `-1.0` constant |
| `adv_ind` | string [1] | tokenized advantage condition: `"positive"` / `"negative"` / `"none"` | initial: `"positive"` for `intervention==1`, `"none"` for `intervention==0`; **rewritten by VLM** for autonomous frames at training time |

Standard LeRobot columns also required (already produced by
`limb convert-lerobot`):

| Column | limb status |
|---|---|
| `observation.state` (14-D YAM bimanual) | ✅ |
| `observation.images.*` (head + 2 wrists) | ✅ |
| `action` (executed; operator's during CORRECTING) | ✅ |
| `task` / `task_index` | ✅ |

Why this fits DAgger data cleanly: limb's CORRECTING phase **is** the
"human-driven / demo-shaped" condition pistar wants for `intervention=1`,
and AUTONOMOUS phase **is** the "rollout / VLM-relabeled" condition for
`intervention=0`. No extra recording needed.

> **Real-robot intuition.** Both pistar (VLM N-step) and RLinf (Critic
> Expert N-step) compute advantage via N-step lookahead. The sim-tuned
> RLinf hyperparameters (`r_fail=-300`, `N=10`) don't apply — pistar uses
> `value_label`/`reward_label` directly, and its VLM learns the right scale
> from the data. Tune at training time in
> `openpi/docs/recap_finetune.md`, not at collection.

---

## What's already recorded (verify, don't rebuild)

Written today by `limb record` with `configs/yam_dagger_pi0_bimanual.yaml` +
`configs/dagger_collection.yaml`. All five pistar columns derive from this:

| Raw file in `recordings/<session>/episode_NNNNNN/` | Used for pistar field(s) |
|---|---|
| `{arm}_actions.npz` (`pos`) | `action` (standard column) |
| `{cam}.mp4` + `{cam}_timestamps.npy` | `observation.images.*` |
| `{arm}_states.npz` | `observation.state` |
| `metadata.json:task_instruction` | `task` |
| `SUCCESS` / `FAILURE` marker | drives `reward` (sparse 1.0 at terminal iff success), `value_label` (ramp vs constant -1) |
| `phase.npy` (`autonomous`/`paused`/`correcting`) | drives `intervention` (1 if "correcting", else 0) and initial `adv_ind` ("positive" if intervention else "none") |
| `interventions.npy` | alternative source for `intervention` (`phase` preferred) |
| `timestamps.npy` | drives `T` (episode length) for `reward_label`/`value_label` formulas |
| `{arm}_policy_actions.npz` | not used by pistar; keep recorded for analysis |

The success/failure marker is written by `DAggerCollectionSession`
(`s` = SUCCESS, SPACE = FAILURE — keyboard trigger).
Code anchors: success path in [dagger_session.py](../limb/recording/dagger_session.py);
phase/policy streams in [episode_recorder.py:296-377](../limb/recording/episode_recorder.py#L296).

**Smoke-test a recorded episode** (must have `SUCCESS`-or-`FAILURE` and
`phase.npy`):

```bash
ls recordings/<session>/episode_000000/
# expect: left_actions.npz right_actions.npz left_states.npz right_states.npz
#         SUCCESS (or FAILURE)  phase.npy  interventions.npy
#         metadata.json  *.mp4  *_timestamps.npy  timestamps.npy
```

---

## What's left to do

`limb convert-lerobot` already emits a v3.0 dataset with `observation.state`,
`observation.images.*`, `action`, `task`, and the optional `phase` /
`correction_index` columns. **Two files change** to emit the five pistar
columns; the recorder/agent code is untouched.

### Task 1 — helpers in `limb/data/episode_utils.py`

Add five functions mirroring `pistar_rlds_demo_processing.py`'s formulas
exactly:

```python
def episode_success(episode_dir: Path) -> bool:
    return (episode_dir / "SUCCESS").exists()

def compute_pistar_intervention(phase: np.ndarray | None,
                                interventions: np.ndarray | None,
                                n_steps: int) -> np.ndarray:
    """1 if 'correcting' phase (operator-driven), else 0."""
    if phase is not None and len(phase) >= n_steps:
        return (np.asarray(phase[:n_steps]) == "correcting").astype(np.int64)
    if interventions is not None and len(interventions) >= n_steps:
        return np.asarray(interventions[:n_steps], dtype=np.int64)
    return np.zeros(n_steps, dtype=np.int64)

def compute_pistar_reward(n_steps: int, success: bool) -> np.ndarray:
    """Sparse: 1.0 at last frame iff success."""
    r = np.zeros(n_steps, dtype=np.float32)
    if success and n_steps > 0:
        r[-1] = 1.0
    return r

def compute_pistar_reward_label(n_steps: int) -> np.ndarray:
    """-1/T per step, 0 at terminal. Same for success and failure."""
    rl = np.full(n_steps, -1.0 / float(n_steps), dtype=np.float32)
    if n_steps > 0:
        rl[-1] = 0.0
    return rl

def compute_pistar_value_label(n_steps: int, success: bool) -> np.ndarray:
    """Initial VLM training target.
       Success: linear ramp -(T-1-t)/T  (≈-1 → 0).
       Failure: constant -1.0 (the VLM refines from here)."""
    if success:
        t = np.arange(n_steps, dtype=np.float32)
        return (-(n_steps - 1 - t) / float(n_steps)).astype(np.float32)
    return np.full(n_steps, -1.0, dtype=np.float32)

def compute_pistar_adv_ind(intervention: np.ndarray) -> list[str]:
    """'positive' on intervention=1 (demo-shaped); 'none' on
    intervention=0 (rewritten later by VLM)."""
    return ["positive" if int(v) == 1 else "none" for v in intervention]
```

### Task 2 — `--pistar` flag in `limb/data/convert_lerobot.py`

Gate the five columns behind a flag so non-pistar datasets stay clean. In
`Args`:

```python
pistar: bool = False
```

In the Phase-2 write loop (around
[convert_lerobot.py:346-354](../limb/data/convert_lerobot.py#L346)),
when `args.pistar` is set, derive and append the columns to `table_data`:

```python
if args.pistar:
    success     = episode_success(episodes[ep_idx])
    interv      = compute_pistar_intervention(res.get("phase"), None, n_steps_out)
    reward      = compute_pistar_reward(n_steps_out, success)
    reward_lbl  = compute_pistar_reward_label(n_steps_out)
    value_lbl   = compute_pistar_value_label(n_steps_out, success)
    adv_ind     = compute_pistar_adv_ind(interv)
    table_data["intervention"] = pa.array(interv,     type=pa.int64())
    table_data["reward"]       = pa.array(reward,     type=pa.float32())
    table_data["reward_label"] = pa.array(reward_lbl, type=pa.float32())
    table_data["value_label"]  = pa.array(value_lbl,  type=pa.float32())
    table_data["adv_ind"]      = pa.array(adv_ind,    type=pa.string())
```

And advertise in `info.json:features` (matching pistar's exact `names`):

```python
if args.pistar:
    features["intervention"] = {"dtype":"int64",  "shape":[1],
                                "names":["intervention_flag"], "fps": info_fps}
    features["reward"]       = {"dtype":"float32","shape":[1],
                                "names":["reward"],       "fps": info_fps}
    features["reward_label"] = {"dtype":"float32","shape":[1],
                                "names":["reward_label"], "fps": info_fps}
    features["value_label"]  = {"dtype":"float32","shape":[1],
                                "names":["value_label"],  "fps": info_fps}
    features["adv_ind"]      = {"dtype":"string", "shape":[1],
                                "names":["adv_ind"],      "fps": info_fps}
```

### Files NOT modified

- `limb/recording/episode_recorder.py` — already writes everything needed.
- `limb/recording/dagger_session.py` — already writes `SUCCESS`/`FAILURE`.
- `limb/agents/dagger/dagger_agent.py` — `policy_pos` is recorded but not
  needed by pistar; leave it.

### Optional column passthrough

- **`policy_action`** — not used by pistar. Keep recorded for offline
  analysis; skip the converter passthrough for v1.
- **`phase` / `correction_index`** — already emitted by the existing
  converter; leave on for debug/visualization. Not required by pistar.

---

## Format / version — verify

- limb emits LeRobot **v3.0**. pistar / openpi's lerobot 0.1.0 reads **v2.1**,
  so run `openpi/scripts/convert_v3_to_v21.py` before training. **The five
  pistar columns survive trivially** — the v3→v2.1 script only symlinks the
  data parquets under new names (`file-NNN.parquet` →
  `episode_NNNNNN.parquet`) and rewrites only the meta files (`info.json`,
  `tasks.jsonl`, `episodes.jsonl`, `episodes_stats.jsonl`). The parquet
  contents (including `adv_ind` as a `string` column) are byte-identical to
  the v3.0 source.
- The five-column schema matches `pistar_rlds_demo_processing.py` exactly,
  so no adapter is needed — pistar's `train.py` / `train_value.py` /
  `label_advantage_from_vlm.py` ingest the dataset directly.

---

## End-to-end usage

### 1. Bootstrap with SFT data

RECAP reshapes a working policy; collect initial demos and SFT first:

```bash
uv run limb record \
  --config-path configs/yam_gello_network_bimanual.yaml \
                configs/collection_pedal.yaml
```

Train pi0.5 SFT per `openpi/docs/yam_finetune.md`; serve it.

### 2. Serve the SFT checkpoint

```bash
cd ~/playground/openpi
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_yam_vial_30fps \
  --policy.dir=ttotmoon/<task>-pi05-v1
```

### 3. Collect RECAP rollout data

The DAgger agent boots in `initial_phase: paused`, so the policy won't move
until you release it.

```bash
uv run limb record \
  --config-path configs/yam_dagger_pi0_bimanual.yaml \
                configs/dagger_collection.yaml
```

Per episode:

1. Stage the scene (optional: right pedal → CORRECTING to back-drive the
   followers to a start pose, right pedal → PAUSED).
2. **SPACE** to start recording (robot still paused).
3. **Left pedal** → AUTONOMOUS; the policy attempts the task.
4. If it drifts: **left pedal** → PAUSED, **right pedal** → CORRECTING,
   bilaterally teleop back on-task, **right pedal** → PAUSED, **left pedal**
   → AUTONOMOUS to hand back.
5. End the episode: **`s`** = success, **SPACE** = failure, **`d`** =
   discard, **`q`** = quit.

Aim for ~50 episodes of mixed outcomes — RECAP needs signal from **both**
successes and failures, so don't filter at collection time.

### 4. Convert with `--pistar`

```bash
uv run limb convert-lerobot \
  --input-dir recordings/<session> \
  --output-dir datasets/<task>_pistar_v1 \
  --target-fps 30 \
  --include-arms left right \
  --pistar           # emit intervention/reward/reward_label/value_label/adv_ind

# Note: DAgger sessions record FOUR arms (followers + leaders). Without
# --include-arms the converter would produce a 28-dim state/action; the
# policy only needs the followers (14-D).

# v3.0 → v2.1 for pistar/openpi
uv run python openpi/scripts/convert_v3_to_v21.py \
  --src=datasets/<task>_pistar_v1 \
  --dst=datasets/<task>_pistar_v1_v21

uv run limb upload \
  --source datasets/<task>_pistar_v1_v21 \
  --target hf://<user>/<task>_pistar_v1
```

### 5. Train (pistar)

See `openpi/docs/recap_finetune.md`. High-level: merge demo + rollout
datasets → `scripts/train_value.py` → `scripts/label_advantage_from_vlm.py`
(rewrites `adv_ind` on rollout frames) → `scripts/train.py` to continue
PiStar fine-tuning with `adv_ind`-conditioned tokens.

### 6. Iterate

Serve the PiStar checkpoint, repeat from step 3 to collect a new rollout
batch, merge it into the dataset, and re-run Stages 5. The VLM relabels
adv_ind for the new rollout frames each round.

---

## Recording dataset hygiene

1. **Keep "boring" autonomous successes.** The value model needs both
   classes; pure-failure data collapses the value distribution, pure-success
   gives no learnable signal.
2. **Don't filter interventions out.** Even short corrections are
   information. `min_correction_steps` is inactive in continuous mode — leave
   it.
3. **One task instruction per session.** The value model and advantage
   quantile are per-task; mixing tasks forces the value model to learn task
   identity too. Separate sessions per task.
4. **Episode count > episode length.** Aim for 20–50 episodes of 30–60 s. The
   value model just needs enough successes/failures to fit.
5. **Label honestly.** The SUCCESS/FAILURE marker drives `reward` and
   `value_label` for the whole episode — a mislabeled episode poisons VLM
   training directly. When unsure, mark FAILURE (SPACE) or discard (`d`).

---

## What this doc does NOT cover

- **VLM value model architecture, advantage labeling, conditioning token,
  serving.** Those live in `openpi/docs/recap_finetune.md` and pistar's
  source (`src/openpi/models/value_model.py`,
  `scripts/label_advantage_from_vlm.py`, `src/openpi/transforms.py`).
- **Reward shaping.** Sparse success reward only; the VLM learns the
  return-shape from `value_label`/`reward_label`. Don't try to encode dense
  rewards at collection.
- **Online RL.** RECAP is *offline*. limb never sees a value model or
  advantage label — the VLM rewrites `adv_ind` at training time.
