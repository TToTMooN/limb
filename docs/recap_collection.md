# RECAP data collection on YAM bimanual

Robot-side recipe for collecting **RECAP** rollout data with limb. The
training side lives in openpi (`openpi/docs/recap_finetune.md`); this doc
covers what limb has to record and convert.

> **TL;DR.** We follow **RLinf's RECAP** (advantage as a CFG conditioning
> token, fine-tuned on the OpenPI pi0.5 — see `recap_finetune.md`). RLinf's
> algorithm is **value-based**: it needs the standard LeRobot columns plus a
> **per-episode success/failure label**. It does *not* require Evo-RL's
> per-frame `policy_action` / `is_intervention` / `collector_policy_id`
> columns. The recording side already captures everything; the only converter
> gap is **exposing `episode_success`**. See [§ What's left to do](#whats-left-to-do).

> **What RECAP is** — *RL with Experience and Corrections via
> Advantage-conditioned Policies*. Four offline stages: (1) compute
> per-trajectory returns from success/failure labels, (2) train a value
> model, (3) compute per-step advantages with N-step lookahead, (4) fine-tune
> the policy, conditioning on advantage. Stages 1–4 are offline; limb owns
> Stage 0 (data).
>
> **References:**
>
> - **Algorithm we use: [RLinf `examples/recap/`](https://github.com/RLinf/RLinf)**
>   ([docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)).
>   Operates on the OpenPI policy model; advantage is a learned **conditioning
>   token** trained with classifier-free guidance. Validated in simulation
>   (LIBERO) only — its hyperparameters (`r_fail=-300`, `N=10`) are sim-tuned
>   and need real-robot recalibration.
> - **Real-robot reference: [MINT-SJTU/Evo-RL](https://github.com/MINT-SJTU/Evo-RL).**
>   The only RECAP validated on real hardware (SO-101, AgileX PiPER). We
>   borrow its collection protocol (human-in-the-loop, `s`/`f` per-episode
>   labels) and real-robot intuition (penalty scaling), but **not** its
>   text-tag conditioning. Its `complementary_info.*` columns are optional
>   extras here, not requirements.

---

## What RECAP needs from rollout data

RLinf's value-based pipeline reads a standard LeRobot dataset plus per-episode
success. The advantage is computed from the value model over observations —
**not** from diffing a recorded `policy_action` against `action` — so the
Evo-RL per-frame extras are optional.

| Column | Needed by RLinf? | limb status |
|---|---|---|
| `observation.state` (14-D) | ✅ required | ✅ recorded + converted |
| `observation.images.*` | ✅ required | ✅ recorded + converted |
| `action` (executed; operator's during CORRECTING) | ✅ required | ✅ recorded + converted |
| `task` (language instruction) | ✅ required | ✅ recorded + converted |
| **`episode_success`** (per-episode bool) | ✅ **required** (Stage 1 returns) | ✅ recorded as `SUCCESS`/`FAILURE` marker; ⚠️ **not yet exposed by converter** |
| `phase` / `correction_index` | optional (analysis, weighting) | ✅ recorded + converted |
| `policy_action` | optional (not used by RLinf) | ✅ recorded (`{arm}_policy_actions.npz`); not converted |
| `is_intervention` | optional (verify advantages skew positive) | ✅ recorded (`interventions.npy`) |
| `complementary_info.*` (Evo-RL names) | not used by RLinf | n/a |

Returns are derived offline by Stage 1 from `episode_success`:

```
r_t = -1                       for every step
terminal = 0                   if episode_success else r_fail
G_t = r_t + γ G_{t+1}          with γ = 1.0
```

limb records **no** rewards or returns — only the per-episode success label.

> **Real-robot penalty.** RLinf's `r_fail = -300` assumes ~300-step LIBERO
> episodes. YAM at 30 Hz / 60 s = 1800 steps, so per-step `-1` sums to `-1800`
> on success and a `-300` failure terminal can't separate the classes. Scale
> `r_fail` to ~5–10× max episode length. Tuned at Stage 1 in
> `openpi/docs/recap_finetune.md`, not at collection.

---

## What's already recorded (verify, don't rebuild)

Written today by `limb record` with `configs/yam_dagger_pi0_bimanual.yaml` +
`configs/dagger_collection.yaml`:

| Raw file in `recordings/<session>/episode_NNNNNN/` | Maps to |
|---|---|
| `{arm}_actions.npz` (`pos`) | `action` (executed) |
| `{cam}.mp4` + `{cam}_timestamps.npy` | `observation.images.*` |
| `{arm}_states.npz` | `observation.state` |
| `metadata.json:task_instruction` | `task` |
| `SUCCESS` / `FAILURE` marker | `episode_success` ← **the one that needs converter exposure** |
| `phase.npy`, `interventions.npy`, `correction_index.npy` | optional metadata |
| `{arm}_policy_actions.npz` | optional (`policy_action`; RLinf doesn't need it) |

The success/failure marker is written by `DAggerCollectionSession`
(`s` = SUCCESS, SPACE = FAILURE — see the keyboard trigger we added). Code
anchors: success path in [dagger_session.py](../limb/recording/dagger_session.py);
phase/policy streams in [episode_recorder.py:296-377](../limb/recording/episode_recorder.py#L296).

**Smoke-test a recorded episode:**

```bash
ls recordings/<session>/episode_000000/
# expect: left_actions.npz right_actions.npz left_states.npz right_states.npz
#         SUCCESS (or FAILURE)  metadata.json  *.mp4  *_timestamps.npy
#         (+ optional: phase.npy interventions.npy *_policy_actions.npz)
```

---

## What's left to do

The LeRobot converter (`limb/data/convert_lerobot.py`, `limb convert-lerobot`)
already emits a v3.0 dataset with `observation.state`, `observation.images.*`,
`action`, `task`, and the optional `phase` / `correction_index` columns. The
**one required gap** for RLinf RECAP is exposing the success label.

### Task 1 — `is_success` per-frame column (required)

**Pinned against the actual RLinf source** (vendored at
`openpi/scripts/recap/compute_returns.py`). Stage 1 reads these columns from
each **data parquet** (`data/chunk-*/file-*.parquet`):

```python
_READ_COLUMNS = ["episode_index", "frame_index", "is_success", "task_index", "task"]
```

and for each episode uses the **last frame's** value:

```python
is_success = bool(is_success_col[ep_end - 1])   # dataset_type="rollout"
```

For `dataset_type="rollout"` (RECAP rollouts) the `is_success` column is
**required** — Stage 1 raises `ValueError` without it. So:

1. Add a helper in `episode_utils.py`:
   `episode_success(episode_dir) -> bool` — `(dir / "SUCCESS").exists()`;
   `FAILURE` or missing → `False`.
2. In `convert_lerobot.py`, write a **per-frame `is_success` bool column** into
   each episode's data-parquet `table_data` (broadcast the episode's single
   label to all its frames — simplest, and the last frame carries the value
   Stage 1 actually reads). Put it next to `episode_index`/`frame_index` at
   [convert_lerobot.py:346-354](../limb/data/convert_lerobot.py#L346):
   ```python
   "is_success": pa.array(np.full(n_steps_out, ep_success, dtype=bool)),
   ```
3. Advertise it in `info.json:features` (dtype `bool`, shape `[1]`), gated like
   the existing `phase`/`correction_index` features so the dataset schema
   stays consistent.

Notes:
- **Column name is `is_success`, not `episode_success`.** That's what RLinf
  reads. (Evo-RL's per-episode `episode_success` is a different convention;
  ignore it here.)
- `task` resolution is already covered: Stage 1 falls back to
  `task_index` → `meta/tasks.jsonl`, which `convert_v3_to_v21.py` produces.
- For the SFT bootstrap dataset you can set `dataset_type: sft` in Stage 1's
  config, which forces `is_success=True` and needs no column — but the DAgger
  rollout dataset must carry `is_success`.

### Task 2 — optional metadata passthrough

Not required by RLinf, but cheap and useful for analysis/verification:

- **`policy_action`** — the converter already *loads* `policy_actions` but
  drops it. If you want it for offline analysis, add a
  `build_policy_action_vector()` and a write block mirroring the `phase`
  handling at [convert_lerobot.py:357-364](../limb/data/convert_lerobot.py#L357).
  **Resample it like `action`** (continuous, ZOH only the gripper dims), not
  like the discrete `phase`. Skip for v1 — RLinf doesn't use it.
- **`is_intervention`** — derive from `phase == "correcting"`. Handy for
  confirming, at Stage 3, that advantages skew positive on intervention
  frames.

### No `--recap` converter / no Evo-RL naming

Since RLinf reads a standard LeRobot dataset + `episode_success`, there's no
need for a separate `convert-recap` script or Evo-RL's `complementary_info.*`
column names. `limb convert-lerobot` (plus Task 1) is the converter. Drop the
earlier plan to mirror Evo-RL's schema — that was for the text-tag path we
abandoned.

---

## Format / version — verify

- limb emits LeRobot **v3.0**. openpi's lerobot 0.1.0 (and RLinf's loaders)
  read **v2.1**, so run `openpi/scripts/convert_v3_to_v21.py` before training.
  Because `is_success` is a **per-frame data-parquet column** (not a meta
  field), the v3→v2.1 converter copies it through with the rest of the row —
  but **verify** it lands in the v2.1 `data/.../episode_*.parquet`:
  ```bash
  uv run python -c "
  import pyarrow.parquet as pq, glob
  f = sorted(glob.glob('<v21_dataset>/data/**/*.parquet', recursive=True))[0]
  print(pq.read_table(f).column_names)"   # expect 'is_success'
  ```
- Stage 1 schema is now pinned (vendored `compute_returns.py`); no adapter
  needed beyond emitting the `is_success` column in Task 1.

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

### 4. Convert and upload

```bash
uv run limb convert-lerobot \
  --input-dir recordings/<session> \
  --output-dir datasets/<task>_recap_v1 \
  --target-fps 30
# (after Task 1, this carries per-episode success)

uv run limb upload \
  --source datasets/<task>_recap_v1 \
  --target hf://<user>/<task>_recap_v1
```

### 5. Train

See `openpi/docs/recap_finetune.md` (RLinf Stages 1–4 on OpenPI).

### 6. Iterate

Serve the RECAP checkpoint, repeat from step 3 tagging the new dataset `v2`.
RLinf keys returns/advantages by `<tag>`, so rounds coexist.

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
5. **Label honestly.** `episode_success` is the *only* required RECAP signal
   limb adds — a mislabeled episode poisons return computation directly. When
   unsure, mark FAILURE (SPACE) or discard (`d`).

---

## What this doc does NOT cover

- **Value model architecture, advantage formula, CFG training, conditioning
  token.** Those live in `openpi/docs/recap_finetune.md` and RLinf's
  `examples/recap/`.
- **Reward shaping.** Sparse terminal reward only; the penalty magnitude is
  tuned at Stage 1, not at collection.
- **Online RL.** RECAP is *offline*. limb never sees a value model or
  advantage label.
