# Policy Server — CLAUDE.md (for companion repo)

## What This Repo Is

A lightweight, standardized policy inference server that wraps any VLA/VLM model
and exposes it over WebSocket for `limb` (the robot control client) to consume.

This is NOT a training framework. It is a thin serving layer.

## Why It Exists

Every VLA framework (OpenPI, LeRobot, Octo, StarVLA, InternVLA, GR00T) has its
own serving code with different protocols, serialization, and conventions. The
robot side (`limb`) shouldn't need a different client for each one.

This repo provides ONE standardized server protocol that can wrap ANY model.
limb talks to this server. That's it.

---

## Protocol Specification

### Transport

- **WebSocket** on configurable `host:port` (default `0.0.0.0:8000`)
- Why WebSocket: simplest bidirectional streaming, works through firewalls,
  no codegen (unlike gRPC), lower latency than REST

### Serialization

- **msgpack** with NumPy array hooks (same as OpenPI)
- NumPy arrays are encoded as:
  ```python
  {"__ndarray__": True, "data": arr.tobytes(), "dtype": str(arr.dtype), "shape": arr.shape}
  ```
- NO pickle (security risk, not language-portable)
- NO JSON for arrays (too slow, too large)

### Connection Handshake

1. Client connects via `ws://host:port`
2. Server immediately sends **metadata** (msgpack):
   ```python
   {
       "model_name": "pi0_fast_base",
       "action_horizon": 25,
       "action_dim": 14,            # e.g. 7 per arm x 2
       "expected_images": ["left_camera", "right_camera", "top_camera"],
       "expected_state_dim": 14,    # joint_pos + gripper per arm
       "image_size": [224, 224],    # expected input resolution
       "accepts_language": True,    # whether model takes text prompts
       "normalization": "server",   # always "server" — client sends raw data
   }
   ```
3. Client uses metadata to configure its observation transforms

### Inference Loop

```
Client (limb)                        Server
  |--- msgpack(observation) -------->|
  |                                  |  transforms → model.infer() → untransform
  |<-- msgpack(response) ------------|
  |         (repeat)                 |
```

### Observation Format (client → server)

```python
{
    # Proprioception (required)
    "state": np.float32 array,           # (state_dim,) — raw joint positions + gripper

    # Images (required, one per camera)
    "images": {
        "left_camera": np.uint8 array,   # (H, W, 3) RGB, resized to image_size by CLIENT
        "right_camera": np.uint8 array,
        "top_camera": np.uint8 array,
    },

    # Language instruction (optional)
    "prompt": "pick up the red cup",

    # Timestamp (optional, for logging)
    "timestamp": 1709561234.567,
}
```

**Client responsibilities:**
- Resize images to `metadata["image_size"]` before sending (saves bandwidth)
- Send raw (unnormalized) proprioceptive state
- Use uint8 for images (not float)

### Response Format (server → client)

```python
{
    # Action chunk (required)
    "actions": np.float32 array,  # (action_horizon, action_dim) — unnormalized, robot-ready

    # Server timing (required, for monitoring)
    "timing": {
        "infer_ms": 45.2,        # model forward pass time
        "total_ms": 48.1,        # including pre/post processing
    },
}
```

**Server responsibilities:**
- Normalize inputs using model's dataset statistics
- Run inference
- Unnormalize outputs back to robot action space
- Return actions in robot-native units (radians, meters, etc.)

### Health Check

- **HTTP GET** `/healthz` on the same port → `200 OK`
- Shares port with WebSocket via the `process_request` hook
- Used by container orchestration, load balancers, and limb's connection retry logic

### Error Handling

- On inference error: server sends msgpack `{"error": "traceback string"}` then closes connection
- Client detects errors by checking for `"error"` key in response

---

## Architecture

```
policy_server/
  server.py              # WebSocket server (main entry point)
  protocol.py            # msgpack serialization with numpy hooks
  health.py              # /healthz endpoint
  base_model.py          # ModelWrapper protocol
  models/
    openpi_wrapper.py    # Wraps OpenPI policy (pi0, pi0-FAST, pi0.5)
    lerobot_wrapper.py   # Wraps LeRobot PreTrainedPolicy (ACT, SmolVLA, etc.)
    octo_wrapper.py      # Wraps Octo JAX model
    hf_vla_wrapper.py    # Wraps HuggingFace VLAs (OpenVLA, InternVLA, etc.)
    custom_wrapper.py    # Template for user's own model
  transforms/
    normalize.py         # Z-score, bounds, quantile normalization
    image.py             # Resize, crop, transpose
    state.py             # State remapping per-embodiment
  configs/
    pi0_yam_bimanual.yaml
    smolvla_yam_bimanual.yaml
    octo_yam_bimanual.yaml
```

### ModelWrapper Protocol

```python
class ModelWrapper(Protocol):
    """Every model adapter implements this."""

    def load(self, checkpoint_path: str, device: str) -> None:
        """Load model weights and normalization stats."""
        ...

    def infer(self, obs: dict) -> np.ndarray:
        """
        Takes raw observation dict (already deserialized from msgpack).
        Returns action chunk: np.float32 (action_horizon, action_dim).
        Handles normalization internally.
        """
        ...

    def metadata(self) -> dict:
        """Returns server metadata dict sent on connection."""
        ...
```

That's the only interface a new model needs to implement.

---

## Config System

YAML configs using OmegaConf (matching limb's convention):

```yaml
# configs/pi0_yam_bimanual.yaml
server:
  host: "0.0.0.0"
  port: 8000

model:
  _target_: policy_server.models.openpi_wrapper.OpenPIWrapper
  config_name: "pi0_fast_base"       # OpenPI config registry name
  checkpoint_dir: "./checkpoints/pi0_fast_base"
  device: "cuda"

# Maps limb's observation keys → model's expected keys
obs_transform:
  state_keys: ["state"]              # already concatenated by client
  image_keys:
    left_camera: "observation/exterior_image_1_left"
    right_camera: "observation/exterior_image_2_left"
    top_camera: "observation/wrist_image_left"
  image_size: [224, 224]

# Maps model output → limb's action format
action_transform:
  action_horizon: 25
  action_dim: 14                     # 7 per arm (6 joints + 1 gripper)
```

---

## Implementation Notes

### Wrapping OpenPI (easiest — ~50 lines)

OpenPI already does everything. The wrapper just:
1. Creates a `TrainedPolicy` from OpenPI's config
2. Calls `policy.infer(obs)` → returns actions
3. Normalization is handled by OpenPI internally

The main value is exposing it through the standardized protocol instead of
OpenPI's own WebSocket server (which has a slightly different wire format).

Alternatively, for OpenPI specifically, limb could just talk to OpenPI's native
server directly — the `openpi_client` package already works. The wrapper is
useful for uniformity but not strictly required.

### Wrapping LeRobot (~100 lines)

```python
from lerobot.policies import make_policy
policy = PolicyClass.from_pretrained(path).to(device)
# policy.select_action(batch) → action tensor
```

LeRobot handles normalization via its PreProcessor/PostProcessor pipeline.
The wrapper batches the observation, runs the pipeline, and returns the chunk.

### Wrapping Octo (~80 lines)

```python
from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
actions = model.sample_actions(obs, task, unnormalization_statistics=...)
```

Octo is JAX so runs in-process. The wrapper handles the observation format
(adding batch/time dimensions, pad masks) and unnormalization stats selection.

### Wrapping HuggingFace VLAs (OpenVLA, InternVLA) (~100 lines)

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
processor = AutoProcessor.from_pretrained(path)
model = AutoModelForVision2Seq.from_pretrained(path)
# processor(image, text) → inputs; model.predict_action(inputs) → action
```

These are typically single-step (no action chunks), so the wrapper may need to
accumulate or just return horizon=1 and let limb handle it.

### Adding a New Model

1. Copy `custom_wrapper.py`
2. Implement `load()`, `infer()`, `metadata()`
3. Add a YAML config pointing to it
4. Done — no protocol code to write

---

## Launch

```bash
# Start server (GPU machine)
uv run policy_server/server.py --config configs/pi0_yam_bimanual.yaml

# Test health
curl http://localhost:8000/healthz

# limb connects automatically when launched with a policy agent config
```

### Docker (recommended for deployment)

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
# Install uv, clone repo, uv sync
EXPOSE 8000
CMD ["uv", "run", "policy_server/server.py", "--config", "configs/pi0_yam_bimanual.yaml"]
```

---

## Dependencies

```
websockets        # WebSocket server
msgpack            # Wire serialization
msgpack-numpy      # NumPy ↔ msgpack hooks (or vendored ~60 lines)
omegaconf          # Config loading
numpy              # Array types
loguru             # Logging (match limb convention)

# Model-specific (install only what you need):
# openpi           # For pi0/pi0.5 models
# lerobot          # For ACT, SmolVLA, DiffusionPolicy
# octo             # For Octo
# transformers     # For HuggingFace VLAs
```

---

## Key Design Decisions

1. **Server-side normalization**: Client sends raw sensor data. Server knows
   the model's normalization stats and handles everything. This means the client
   (limb) doesn't need to know anything about the model's training data.

2. **msgpack not pickle**: Pickle is a security risk and Python-only. msgpack
   is fast, compact, and has libraries in every language. The NumPy hooks add
   ~60 lines of code.

3. **WebSocket not gRPC**: gRPC requires protobuf codegen and is overkill for
   a single `infer(obs) → actions` RPC. WebSocket is simpler and just as fast
   for our use case (one client, LAN, ~20-50 Hz).

4. **Metadata on connect**: The server tells the client what it expects (image
   size, state dim, action shape). This means limb can auto-configure its
   observation transforms without hardcoding per-model knowledge.

5. **One model per server instance**: Keep it simple. Run multiple server
   processes for multiple models. Container orchestration handles routing.

6. **YAML config with `_target_`**: Same pattern as limb. Familiar, minimal,
   no framework lock-in.

---

## Development Conventions

- **Package manager**: `uv`
- **Python**: 3.11
- **Linter**: `ruff` (line length 119)
- **Logging**: `loguru`
- **Config**: OmegaConf (same as limb)
- **Testing**: pytest with mock WebSocket connections

---

## What This Repo Does NOT Do

- Training or fine-tuning (use upstream frameworks)
- Dataset management (use LeRobot datasets or your own)
- Robot control (that's limb)
- Camera drivers (that's limb)
- Action chunking/smoothing on the client side (that's limb)
- Multi-model serving / model routing (use separate instances + a load balancer)
