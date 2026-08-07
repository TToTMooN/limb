# pi0.6 / PiStar — train + serve runbook (YAM, Plan C)

Concrete copy-paste commands to **train pistar (pi0.6) on a YAM `--pistar`
dataset and serve it back to limb**, skipping pistar Stages 4–5 (VLM value
model + VLM-based advantage labeling).

Picks up from [`docs/dagger_for_pi06.md`](dagger_for_pi06.md) — collect →
convert → v2.1 done; dataset at `datasets/<task>_pistar_v1_v21/`.

> **Why Plan C** (skip Stages 4–5). pistar's full RECAP loop trains a VLM
> value model (Stage 4) and uses it to relabel `adv_ind` on autonomous
> frames (Stage 5). The VLM base weights are distributed via AliPan
> (Chinese cloud, often inaccessible) **and** pistar's
> `scripts/train_value.py` currently has a broken import
> (`ValueModelWeightLoader` is referenced but not defined in
> `weight_loaders.py` on `main`). Plan C sidesteps both blockers by going
> straight to Stage 6 with the **limb-supplied initial `adv_ind`**:
> `"positive"` on CORRECTING (intervention=1) frames, `"none"` on
> autonomous frames. The autonomous-success signal is wasted (vs full
> RECAP), but on a small dataset the value model would heavily overfit
> anyway — this lets you validate the entire pi0.6 path end-to-end first.

For the algorithm / why, read [`openpi/docs/recap_finetune.md`](../openpi/docs/recap_finetune.md).
For the upstream pistar source, see [ybpy/pistar](https://github.com/ybpy/pistar).

---

## Layout

pistar runs as a **sibling** to openpi under `limb/`. openpi/YAM-SFT
(`yam_finetune.md`) stays untouched; pistar handles pi0.6 RECAP.

```
limb/
├── openpi/                              # YAM-SFT path — unchanged
├── pistar/                              # ybpy/pistar (cloned + YAM patched)
│   ├── src/openpi/policies/aloha_policy.py        ← YAM patch: adv_ind passthrough
│   ├── src/openpi/policies/value_policy.py        ← YAM patch: head/wrist camera names
│   ├── src/openpi/training/config.py
│   │   ├── LeRobotAlohaDataConfig                 ← YAM patch: adv_ind_dropout plumbing
│   │   ├── TrainConfig "pi06_yam_vial_30fps"      ← NEW
│   │   └── TrainConfig "pi06_yam_vial_30fps_infer" ← NEW
└── datasets/
    └── vial_rollout_v1_v21/             # v2.1 from --pistar + convert_v3_to_v21
```

## Total pistar-side changes for Plan C

| File | Change | Lines |
|---|---|---|
| `pistar/src/openpi/policies/aloha_policy.py` | Add `if "adv_ind" in data: inputs["adv_ind"] = data["adv_ind"]` at end of `AlohaInputs.__call__` | +6 |
| `pistar/src/openpi/policies/value_policy.py` | Add YAM column names to `base_candidates` / `wrist_candidates` | +10 |
| `pistar/src/openpi/training/config.py` | `LeRobotAlohaDataConfig`: add `adv_ind_dropout: bool = True` field, plumb to `ModelTransformFactory` | +7 |
| `pistar/src/openpi/training/config.py` | Four new TrainConfigs: `pi06_yam_vial_30fps{,_infer,_lora,_lora_infer}` | +190 |

Total: ~110 lines added to pistar. Everything else is the upstream pistar code.

---

## Prerequisites

1. **pistar cloned + installed in its own venv** (it is itself an openpi fork
   — do NOT share a venv with `limb/openpi`):

   ```bash
   cd ~/Desktop/research/limb/pistar
   git submodule update --init --recursive

   uv venv --python 3.11.9 ~/.venvs/pistar
   source ~/.venvs/pistar/bin/activate
   GIT_LFS_SKIP_SMUDGE=1 uv sync --active
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   uv pip install -r pistar_requirements.txt
   ```

2. **pi05_base weights** (publicly hosted on GCS, no AliPan needed):

   ```bash
   # Option A: let pistar/openpi pull from gs:// at first training step
   gcloud auth application-default login

   # Option B: pre-download (faster, also works without auth)
   mkdir -p ~/pi05_base
   gsutil -m rsync -r gs://openpi-assets/checkpoints/pi05_base ~/pi05_base
   # Then in pistar's config.py, change the CheckpointWeightLoader path for
   # pi06_yam_vial_30fps to "/home/<user>/pi05_base/params"
   ```

3. **A YAM `--pistar` v2.1 dataset.** If not yet:

   ```bash
   # from limb's venv (not pistar's), in limb/
   uv run limb convert-lerobot \
     --input-dir recordings/<your_session> \
     --output-dir datasets/<task>_pistar_v1 \
     --target-fps 30 --include-arms left right --pistar \
     --task "<your task instruction>"

   uv run python openpi/scripts/convert_v3_to_v21.py \
     --src=datasets/<task>_pistar_v1 \
     --dst=datasets/<task>_pistar_v1_v21
   ```

   The reference dataset `datasets/vial_rollout_v1_v21/` (10 episodes,
   21286 frames @ 30 Hz) is already there.

4. **Symlink the v2.1 dataset into pistar's lerobot cache** — **REQUIRED.**
   pistar resolves `repo_id="local/<name>"` via
   `~/.cache/huggingface/lerobot/local/<name>/meta/info.json`. Without this
   symlink the loader falls through to HuggingFace Hub and 404s with a
   misleading `RepositoryNotFoundError`.

   ```bash
   mkdir -p ~/.cache/huggingface/lerobot/local
   ln -sfn /home/ssc/Desktop/research/limb/datasets/vial_rollout_v1_v21 \
          ~/.cache/huggingface/lerobot/local/vial_rollout_v1_v21

   # verify
   ls ~/.cache/huggingface/lerobot/local/vial_rollout_v1_v21/meta/info.json
   # → should print the path; if not, the symlink is broken or the dataset is missing
   ```

---

## Train (Stage 6 directly from pi05_base)

The YAM training configs in pistar are patterned on the openpi-YAM SFT
(`pi05_yam_vial_30fps`) with `pistar=True` added so the tokenizer ingests
`adv_ind`. The dataset already carries `adv_ind` per frame (limb-supplied:
positive on CORRECTING / none on autonomous).

### Pick the config that fits your starting checkpoint × your GPU

| Config | Starts from | Trains | Memory | Use when |
|---|---|---|---|---|
| `pi06_yam_vial_30fps_lora_from_sft` | **Your YAM-task SFT** (`/home/ssc/checkpoints/yam-vial-place-pi05-v1/params`) | LoRA adapters only | ≈ 16–20 GB | Single 24 GB GPU + you already SFT'd pi0.5 on the task — **RECOMMENDED** |
| `pi06_yam_vial_30fps_lora` | `pi05_base` (generic) | LoRA adapters only | ≈ 16–20 GB | Single 24 GB GPU + no task-specific SFT |
| `pi06_yam_vial_30fps` | `pi05_base` | All ~3B params | ≥ 80 GB | Multi-GPU rig or H100/A100-80GB |
| (full from SFT) | omitted | — | — | Build one yourself if needed |

Why `_from_sft` is the right starting point when you have it: pistar's
canonical pipeline is *SFT first, then PiStar fine-tune on top*. Starting
the LoRA fine-tune from a YAM-task SFT gives the policy a head start on
the action distribution; pi0.6 then only has to learn the
advantage-conditioning token (`adv_ind`). Starting from `pi05_base`
forces it to learn both the task *and* the token in 5k steps — harder on
a small dataset.

> **Note on the checkpoint path.** `pi06_yam_vial_30fps_lora_from_sft`
> hardcodes `/home/ssc/checkpoints/yam-vial-place-pi05-v1/params`. If your
> machine has it elsewhere, edit the `weight_loader` in
> [`pistar/src/openpi/training/config.py`](../pistar/src/openpi/training/config.py)
> or download from
> [HF](https://huggingface.co/ttotmoon/yam-vial-place-pi05-v1) into that path:
> ```bash
> huggingface-cli download ttotmoon/yam-vial-place-pi05-v1 \
>   --local-dir /home/ssc/checkpoints/yam-vial-place-pi05-v1
> ```

For your RTX 5090 Laptop (24 GB) + the YAM SFT checkpoint you have:

```bash
source ~/.venvs/pistar/bin/activate
cd ~/Desktop/research/limb/pistar

XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py pi06_yam_vial_30fps_lora_from_sft \
    --exp-name=plan_c_v0 \
    --overwrite
```

What changed vs the full config:

- `paligemma_variant="gemma_2b_lora"`, `action_expert_variant="gemma_300m_lora"` — turns on LoRA adapters in the model.
- `freeze_filter=...get_freeze_filter()` — freezes the base weights; only LoRA params get optimizer state.
- `ema_decay=None` — EMA off (matches the pi0_aloha_lora pattern in upstream openpi).
- `batch_size=4` — tuned for 24 GB. Bump to 8 if you have headroom; back off to 2 if it still OOMs.
- `num_workers=4` — lighter than the full config's 8 since LoRA spends less time waiting on the GPU.

`num_train_steps=5_000` is preserved. On your 21k-frame dataset that's ~5
epochs at batch 4 — fine for an initial run.

CLI flags:

| Flag | Notes |
|---|---|
| `--exp-name=plan_c_v0` | Sub-directory under checkpoints. Pick something descriptive. |
| `--overwrite` | Wipe a previous run with the same `exp-name`. Use `--resume` to continue. |
| `XLA_PYTHON_CLIENT_*` | Pi05 wants the whole GPU; pistar's README sets these. |

Checkpoints land at:
```
~/Desktop/research/limb/pistar/checkpoints/pi06_yam_vial_30fps/plan_c_v0/<step>/
```

**Healthy signs during training:**
- Loss falls from ~7–9 (initial) toward ~1–2 within the first ~500 steps.
- No NaN/Inf in the log.
- `adv_ind` token stats logged periodically (positive vs none mix matches
  your dataset's intervention rate — for the reference dataset, ~33%
  positive).

---

## Serve back to limb

```bash
source ~/.venvs/pistar/bin/activate
cd ~/Desktop/research/limb/pistar

# LoRA from SFT → match with the _from_sft_infer config:
python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi06_yam_vial_30fps_lora_from_sft_infer \
  --policy.dir=checkpoints/pi06_yam_vial_30fps_lora_from_sft/plan_c_v0/4999

# LoRA from pi05_base:
# python scripts/serve_policy.py --port=8111 policy:checkpoint \
#   --policy.config=pi06_yam_vial_30fps_lora_infer \
#   --policy.dir=checkpoints/pi06_yam_vial_30fps_lora/plan_c_v0/4999

# Full fine-tune (multi-GPU only):
# python scripts/serve_policy.py --port=8111 policy:checkpoint \
#   --policy.config=pi06_yam_vial_30fps_infer \
#   --policy.dir=checkpoints/pi06_yam_vial_30fps/plan_c_v0/4999
```

- The infer config MUST match the training config's model variant (LoRA ↔
  LoRA, full ↔ full, _from_sft ↔ _from_sft). Mismatched LoRA-vs-full
  restore fails because the param tree shapes differ.
- Each `_infer` variant uses `adv_ind_dropout=False` so the positive tag is
  always present at serving; everything else mirrors the training config.
- `--policy.dir` points at one specific step directory.

The advantage conditioning is **entirely inside the openpi tokenizer**;
limb's `OpenPIClient` does not see `adv_ind`. The token gets inserted by
the tokenizer because `model_config.pistar=True`. limb's prompt stays the
plain task instruction.

---

## limb side — no changes

The DAgger deployment config (`configs/yam_dagger_pi0_bimanual.yaml`) is
unchanged; just point at the new server. In the config:

```yaml
inner_policy:
  client:
    _target_: limb.agents.policy_learning.policy_client.OpenPIClient
    host: "0.0.0.0"
    port: 8111
  obs_transform:
    prompt: "Use one arm to grasp the papercup and hand it over to the other arm"
    # ↑ the SAME plain task instruction you trained on. No "Advantage:" tag.
    image_keys:
      cam_high: "head_camera-images-rgb"
      cam_left_wrist: "left_wrist_camera-images-rgb"
      cam_right_wrist: "right_wrist_camera-images-rgb"
    image_size: [224, 224]
    state_keys: ["left-joint_pos", "left-gripper_pos",
                 "right-joint_pos", "right-gripper_pos"]
```

Launch deployment the same way as the SFT path:

```bash
# from limb/, in limb venv
uv run limb teleop --config-path configs/yam_dagger_pi0_bimanual.yaml
# or
uv run limb teleop --config-path configs/yam_pi0_bimanual.yaml
```

---

## What you give up vs full RECAP

In Plan C, **autonomous frames are labeled `"none"`** — they're neutral
training samples. In full RECAP (Stages 4–5), those frames would be
classified `"positive"` or `"negative"` by the VLM. Concretely:

- The policy learns to imitate **intervention (correction)** frames
  strongly (positive condition).
- It treats autonomous frames as neutral context — they pull the model
  toward their observed action distribution, but without an explicit
  good/bad signal.

This is essentially "**DAgger-imitation with a learned positive-correction
token**", not classifier-free guidance over advantage. It still gives you:

- The correct conditioning channel (`pistar=True` → `adv_ind` token).
- A checkpoint that can be re-trained against a real VLM advantage labeling
  later without changing the policy architecture or the serving path.

When you eventually get the VLM base weights, you can re-run the full
Stage 4 → 5 → 6 on this same dataset and the resulting checkpoint will be a
drop-in replacement (same architecture, same serving config name).

---

## When you do get the VLM weights — going to full RECAP

Stage 4 (train_value.py) and Stage 5 (label_advantage_from_vlm.py) commands
were drafted earlier and live in the git history of this doc if needed; the
short version is:

```bash
# Stage 4 — VLM value model SFT
python scripts/train_value.py \
  --data_dir <…>/datasets/vial_rollout_v1_v21 \
  --checkpoint_dir ~/pistar_ckpts/yam_value_v1 \
  --tokenizer_path ~/pistar_assets/tokenizer.model \
  --load_pretrained \
  --batch_size 32 --num_train_steps 30000

# Stage 5 — relabel adv_ind on intervention=0 frames in place
python scripts/label_advantage_from_vlm.py \
  --data_dir <…>/datasets/vial_rollout_v1_v21 \
  --checkpoint_dir ~/pistar_ckpts/yam_value_v1 \
  --tokenizer_path ~/pistar_assets/tokenizer.model \
  --batch_size 8 --lookahead 50 --use_ema \
  --base_image_col   observation.images.head_camera \
  --wrist_image_col  observation.images.left_wrist_camera \
  --right_wrist_image_col observation.images.right_wrist_camera

# Stage 6 — re-run THE SAME training command above; pistar reads the
# updated adv_ind from the dataset automatically.
python scripts/train.py pi06_yam_vial_30fps --exp-name=recap_v1 --overwrite
```

Two known prerequisites for Stage 4:

1. **VLM base weights + Gemma3 tokenizer.model** — from pistar's AliPan link
   in `pistar/README.md`, or from public Google sources (SigLIP from
   `gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz`, Gemma3-270m
   from HuggingFace `google/gemma-3-270m`).
2. **Patch `ValueModelWeightLoader`** — pistar's `train_value.py` imports
   `ValueModelWeightLoader` from `weight_loaders.py`, but that class is
   missing on current `main`. Define it by mirroring `PaliGemmaWeightLoader`.

---

## Common issues

**`KeyError: 'adv_ind'` during training**
The Aloha repack isn't passing `adv_ind` through. Verify:
```bash
grep -A 12 'pi06_yam_vial_30fps' src/openpi/training/config.py | grep adv_ind
# expect: "adv_ind": "adv_ind"
```
If empty, the YAM TrainConfig isn't installed correctly.

**Model trains but `adv_ind` token has no effect at inference**
`adv_ind_input` not wired. Verify:
```bash
grep -n "adv_ind_input=model_config.pistar" src/openpi/training/config.py
# expect: a hit in ModelTransformFactory.__call__
```

**`XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory` at `init_train_state`**
The full `pi06_yam_vial_30fps` config needs ~80 GB just to allocate AdamW
state for ~3B params. Switch to the LoRA config
`pi06_yam_vial_30fps_lora` (5% trainable, fits in 24 GB). If LoRA at
`batch_size=4` still OOMs, edit the config to `batch_size=2`. If still
OOM, restart any process holding GPU memory (`nvidia-smi`) and try again.

**Loss diverges / NaN after a few steps**
- Drop `batch_size` further (edit the TrainConfig).
- Check the dataset's `value_label` and `reward_label` distributions for
  `inf`/`NaN`; should be `[-1, 0]` real-valued.

**Serving fails with `shape mismatch` / param tree errors**
The `--policy.config` model variant must match the trained checkpoint's
variant. LoRA-trained → `pi06_yam_vial_30fps_lora_infer`. Full-trained →
`pi06_yam_vial_30fps_infer`. You cannot restore one variant's checkpoint
with the other's config.

**`serve_policy.py` fails to find the checkpoint**
`--policy.dir` must point at a single step directory (e.g.
`.../plan_c_v0/4999`), not the parent run directory.

---

## See also

- [`docs/dagger_for_pi06.md`](dagger_for_pi06.md) — collection + convert quickstart
- [`docs/recap_collection.md`](recap_collection.md) — schema + DAgger-to-pistar field mapping
- [`openpi/docs/recap_finetune.md`](../openpi/docs/recap_finetune.md) — pi0.6 RECAP algorithm + full pipeline
- [`openpi/docs/yam_finetune.md`](../openpi/docs/yam_finetune.md) — the openpi-YAM SFT this config mirrors
- [ybpy/pistar `README.md`](https://github.com/ybpy/pistar) — upstream pistar reference
