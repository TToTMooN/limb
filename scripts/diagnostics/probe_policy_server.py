"""Probe a lobe-serve policy with TRAINING-dataset frames.

Connects to the running policy server (default ws://0.0.0.0:8000) and sends
observations built directly from the dataset that the policy was trained on.
Compares the policy's returned action chunks against the ground-truth recorded
actions for the same frames.

If the policy outputs sensible chunks for training-dataset frames, the bug is
in what limb sends (image format, state layout). If it returns the same noisy
chunks even here, the bug is server-side (model weights, normalization stats,
preprocessor wiring).

Default dataset: ``ttotmoon/place_the_vial_into_the_stand_1to4`` (the one the
yam-place-vial-fm-v0 checkpoint was trained on).

Usage::

    uv run scripts/diagnostics/probe_policy_server.py \\
        --host 0.0.0.0 --port 8000 \\
        --parquet /tmp/yam_vial.parquet \\
        --head-video /tmp/probe_data/head_ep0.mp4 \\
        --left-wrist-video /tmp/probe_data/left_wrist_ep0.mp4 \\
        --right-wrist-video /tmp/probe_data/right_wrist_ep0.mp4 \\
        --frames 8 --prompt "place the vial into the stand"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import tyro
from loguru import logger


@dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    parquet: str = "/tmp/yam_vial.parquet"  # produced by curl in probe instructions
    head_video: str = "/tmp/probe_data/head_ep0.mp4"
    left_wrist_video: str = "/tmp/probe_data/left_wrist_ep0.mp4"
    right_wrist_video: str = "/tmp/probe_data/right_wrist_ep0.mp4"
    frames: int = 10  # how many sequential frames to probe with
    start_frame: int = 0  # which frame in the episode to start at
    image_size: tuple[int, int] = (240, 320)  # (H, W) — must match policy training
    prompt: str = "place the vial into the stand"
    save_path: Optional[str] = "/tmp/probe_trace.npz"


def _read_video_frames(path: str, n: int, start: int = 0) -> np.ndarray:
    """Read `n` consecutive frames starting from frame `start`, return (n, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return np.stack(frames)


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def main(args: Args) -> None:
    """Probe the policy server with training-dataset frames and compare against GT actions."""
    import pandas as pd
    from websockets.sync.client import connect

    encode = msgpack_numpy.encode
    decode = msgpack_numpy.decode

    logger.info("Loading training parquet: {}", args.parquet)
    df = pd.read_parquet(args.parquet)
    ep0 = df[df["episode_index"] == df["episode_index"].iloc[0]].reset_index(drop=True)
    logger.info("Episode 0 has {} frames", len(ep0))

    f0 = args.start_frame
    f1 = f0 + args.frames
    states = np.stack(ep0["observation.state"].values[f0:f1]).astype(np.float32)  # (n, 14)
    gt_actions = np.stack(ep0["action"].values[f0:f1]).astype(np.float32)  # (n, 14)

    logger.info("Reading videos (frames {} -> {})...", f0, f1)
    head = _read_video_frames(args.head_video, args.frames, f0)
    lwrist = _read_video_frames(args.left_wrist_video, args.frames, f0)
    rwrist = _read_video_frames(args.right_wrist_video, args.frames, f0)
    logger.info("Videos: head={} left={} right={}", head.shape, lwrist.shape, rwrist.shape)

    H, W = args.image_size
    head_r = np.stack([_resize(f, H, W) for f in head])
    lw_r = np.stack([_resize(f, H, W) for f in lwrist])
    rw_r = np.stack([_resize(f, H, W) for f in rwrist])

    uri = f"ws://{args.host}:{args.port}"
    logger.info("Connecting to {}", uri)
    ws = connect(uri, open_timeout=10.0, max_size=None)
    raw = ws.recv()
    metadata = msgpack.unpackb(raw, raw=False, object_hook=decode)
    logger.info("Server metadata: {}", metadata)

    horizon = metadata.get("action_horizon", 8)
    action_dim = metadata.get("action_dim", 14)

    all_chunks: List[np.ndarray] = []
    all_dts: List[float] = []
    for i in range(args.frames):
        obs = {
            "state": states[i],
            "images": {
                "head_camera": head_r[i],
                "left_wrist_camera": lw_r[i],
                "right_wrist_camera": rw_r[i],
            },
            "prompt": args.prompt,
        }
        t0 = time.time()
        ws.send(msgpack.packb(obs, use_bin_type=True, default=encode))
        raw = ws.recv()
        dt = time.time() - t0
        if isinstance(raw, str):
            raise RuntimeError(f"server error: {raw}")
        resp: Dict[str, Any] = msgpack.unpackb(raw, raw=False, object_hook=decode)
        if "error" in resp:
            raise RuntimeError(f"server error: {resp['error']}")
        actions = resp["actions"]  # (horizon, action_dim)
        all_chunks.append(actions)
        all_dts.append(dt)
        logger.info(
            "frame {:>3d}  dt={:.3f}s  chunk shape={}  chunk[0]={}",
            i, dt, actions.shape, np.round(actions[0], 3).tolist(),
        )

    ws.close()

    chunks = np.stack(all_chunks)  # (n_frames, horizon, action_dim)
    dts = np.array(all_dts)
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON: policy output vs ground-truth actions")
    logger.info("=" * 70)

    arm_names = [
        "left_j0", "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_grip",
        "right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_grip",
    ]

    # 1) For each frame, compare chunk[0] to ground-truth action at that frame
    logger.info("")
    logger.info("--- chunk[0] vs GT action (per-frame, per-dim) ---")
    for f in range(args.frames):
        delta = chunks[f, 0] - gt_actions[f]
        logger.info(
            "frame {:>3d}  state={}  GT_action={}  pred_action={}  |err|_max={:.3f}",
            f,
            np.round(states[f][:7], 3).tolist(),
            np.round(gt_actions[f][:7], 3).tolist(),
            np.round(chunks[f, 0][:7], 3).tolist(),
            np.abs(delta).max(),
        )

    # 2) Within-chunk smoothness — what we saw was ~0.3-0.8 rad mean delta
    logger.info("")
    logger.info("--- within-chunk |Δ|/step (mean across all chunks) ---")
    within = np.abs(np.diff(chunks, axis=1)).mean(axis=(0, 1))  # (action_dim,)
    for i, n in enumerate(arm_names):
        logger.info("  {:<10s}  mean|Δ|={:.4f}  max|Δ|={:.4f}", n, within[i], np.abs(np.diff(chunks, axis=1))[..., i].max())

    # 3) Ground-truth within-chunk smoothness (the floor)
    gt_within = np.abs(np.diff(gt_actions, axis=0)).mean(axis=0)
    logger.info("")
    logger.info("--- GT-action |Δ|/step at 30 Hz (the reference smoothness) ---")
    for i, n in enumerate(arm_names):
        logger.info("  {:<10s}  mean|Δ|={:.4f}  max|Δ|={:.4f}", n, gt_within[i], np.abs(np.diff(gt_actions, axis=0))[:, i].max())

    # 4) "Best-case overlap" — do chunks at frame f match GT actions at f..f+horizon-1?
    logger.info("")
    logger.info("--- chunk vs GT trajectory: predicted_chunk[t] vs GT_action[frame+t] ---")
    h = min(horizon, len(gt_actions))
    for f in range(min(3, args.frames)):
        for t in range(min(h, args.frames - f)):
            err = chunks[f, t] - gt_actions[f + t]
            logger.info(
                "  frame={} t={} | pred[:7]={} | GT[:7]={} | |err|max={:.3f}",
                f, t, np.round(chunks[f, t][:7], 3).tolist(),
                np.round(gt_actions[f + t][:7], 3).tolist(),
                np.abs(err).max(),
            )

    logger.info("")
    logger.info("Latency: mean={:.3f}s, p95={:.3f}s, max={:.3f}s", dts.mean(), np.quantile(dts, 0.95), dts.max())

    if args.save_path:
        np.savez(
            args.save_path,
            chunks=chunks,
            gt_actions=gt_actions,
            states=states,
            dts=dts,
            metadata=str(metadata),
        )
        logger.info("Saved trace: {}", args.save_path)

    # Verdict
    logger.info("")
    logger.info("=" * 70)
    logger.info("VERDICT")
    logger.info("=" * 70)
    chunk0_err = np.abs(chunks[:, 0] - gt_actions).max()
    if chunk0_err < 0.05:
        logger.info("✓ Policy chunk[0] closely tracks GT action — server is operating correctly.")
    elif chunk0_err < 0.3:
        logger.info("△ Policy chunk[0] drifts from GT by up to {:.2f} rad — degraded but reasonable.", chunk0_err)
    else:
        logger.warning(
            "✗ Policy chunk[0] differs from GT by up to {:.2f} rad — server is producing "
            "wrong actions even for training-dataset obs. Investigate normalization/checkpoint/preprocessor.",
            chunk0_err,
        )
    within_smooth = within[:6].max()
    if within_smooth < 0.1:
        logger.info("✓ Within-chunk actions are smooth (max joint Δ={:.3f} rad/step).", within_smooth)
    else:
        logger.warning(
            "✗ Within-chunk joint Δ up to {:.2f} rad/step — chunks are not coherent temporal trajectories.",
            within_smooth,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
