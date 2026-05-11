"""Probe a pi0/pi0.5/pi0-FAST policy server (OpenPI native protocol) with
training-dataset frames. Companion to probe_policy_server.py (which uses the
lobe-msgpack protocol). Sends real state + 3 wrist/head frames + prompt from
a recorded episode, compares each returned chunk against the ground-truth
recorded action.

Useful for telling apart "policy is bad" from "limb is sending obs wrong".
If chunk[0] tracks GT actions within a few mrad, the policy is healthy and
any deployment issue is client-side.

Uses the vendored OpenPI client in ``limb.vendor.openpi_client`` — no
external dependency.

Usage::

    uv run scripts/diagnostics/probe_openpi_server.py \\
        --host 0.0.0.0 --port 8111 \\
        --parquet /tmp/yam_vial.parquet \\
        --head-video  /tmp/probe_data/head_ep0.mp4 \\
        --left-wrist-video  /tmp/probe_data/left_wrist_ep0.mp4 \\
        --right-wrist-video /tmp/probe_data/right_wrist_ep0.mp4 \\
        --frames 30 --prompt "place the vial into the stand"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import tyro
from loguru import logger


@dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8111
    parquet: str = "/tmp/yam_vial.parquet"
    head_video: str = "/tmp/probe_data/head_ep0.mp4"
    left_wrist_video: str = "/tmp/probe_data/left_wrist_ep0.mp4"
    right_wrist_video: str = "/tmp/probe_data/right_wrist_ep0.mp4"
    frames: int = 30                     # samples spread across the episode
    image_size: tuple[int, int] = (224, 224)  # AlohaInputs default
    prompt: str = "place the vial into the stand"
    save_path: Optional[str] = "/tmp/openpi_probe.npz"


def _read_at(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {idx}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_chw(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))


def main(args: Args) -> None:
    """Send training-dataset obs to an OpenPI server and compare returned chunks vs GT actions."""
    import pandas as pd

    from limb.vendor.openpi_client import WebsocketClientPolicy

    df = pd.read_parquet(args.parquet)
    ep0 = df[df["episode_index"] == df["episode_index"].iloc[0]].reset_index(drop=True)
    states = np.stack(ep0["observation.state"].values).astype(np.float32)
    actions = np.stack(ep0["action"].values).astype(np.float32)
    n_total = len(actions)
    logger.info("Episode 0: {} frames", n_total)

    H, W = args.image_size
    dim_names = [
        "left_j0", "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_grip",
        "right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_grip",
    ]
    # Leave a 50-frame margin at the end so chunk[t] vs GT[frame+t] has GT to compare against.
    sample_idxs = np.linspace(0, max(0, n_total - 50 - 1), args.frames, dtype=int)
    logger.info("Sampling {} frames", args.frames)

    caps = [cv2.VideoCapture(p) for p in (args.head_video, args.left_wrist_video, args.right_wrist_video)]
    head, lw, rw = [], [], []
    for idx in sample_idxs:
        head.append(_resize_chw(_read_at(caps[0], int(idx)), H, W))
        lw.append(_resize_chw(_read_at(caps[1], int(idx)), H, W))
        rw.append(_resize_chw(_read_at(caps[2], int(idx)), H, W))
    for c in caps:
        c.release()

    logger.info("Connecting to ws://{}:{}", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    logger.info("Server metadata: {}", client.get_server_metadata())

    chunks, dts = [], []
    for i, idx in enumerate(sample_idxs):
        obs = {
            "state": states[idx],
            "images": {
                "cam_high": head[i],
                "cam_left_wrist": lw[i],
                "cam_right_wrist": rw[i],
            },
            "prompt": args.prompt,
        }
        t0 = time.time()
        resp = client.infer(obs)
        dts.append(time.time() - t0)
        chunks.append(np.asarray(resp["actions"]))
        if i < 3 or i % 5 == 0:
            logger.info(
                "frame {:>4d}  dt={:.3f}s  shape={}  chunk[0][:7]={}  GT[:7]={}",
                int(idx), dts[-1], chunks[-1].shape,
                np.round(chunks[-1][0, :7], 3).tolist(),
                np.round(actions[idx, :7], 3).tolist(),
            )

    chunks = np.stack(chunks)
    gts = actions[sample_idxs]
    horizon = chunks.shape[1]
    logger.info("Chunk shape: {} — horizon {} dim {}", chunks.shape, horizon, chunks.shape[-1])
    logger.info("Latency steady-state: median={:.3f}s, p95={:.3f}s, max={:.3f}s",
                float(np.median(dts)), float(np.quantile(dts, 0.95)), float(np.max(dts)))

    # chunk[0] vs GT[frame]
    err0 = np.abs(chunks[:, 0] - gts)
    logger.info("")
    logger.info("=" * 72)
    logger.info("chunk[0] vs GT[frame] (across {} frames)", args.frames)
    logger.info("=" * 72)
    logger.info("  MSE        = {:.6f}",     (err0 ** 2).mean())
    logger.info("  RMSE       = {:.6f}", np.sqrt((err0 ** 2).mean()))
    logger.info("  mean|err|  = {:.6f}",     err0.mean())
    logger.info("  max|err|   = {:.6f}",     err0.max())
    logger.info("  p50/p95/p99 = {:.4f} / {:.4f} / {:.4f}",
                np.quantile(err0, 0.5), np.quantile(err0, 0.95), np.quantile(err0, 0.99))
    logger.info("")
    logger.info("per-dim mean|err|:")
    for i, n in enumerate(dim_names):
        logger.info("  {:<12s}  mean={:.4f}  max={:.4f}", n, err0[:, i].mean(), err0[:, i].max())

    # chunk[t] vs GT[frame+t] across the full horizon (chunk lookahead fidelity)
    traj_errs = []
    for i, idx in enumerate(sample_idxs):
        end = min(int(idx) + horizon, n_total)
        h_use = end - int(idx)
        traj_errs.append(np.abs(chunks[i, :h_use] - actions[int(idx):int(idx) + h_use]))
    traj_errs = np.concatenate(traj_errs, axis=0)
    logger.info("")
    logger.info("=" * 72)
    logger.info("chunk[t] vs GT[frame+t] across full horizon ({} predictions)", traj_errs.shape[0])
    logger.info("=" * 72)
    logger.info("  MSE        = {:.6f}",     (traj_errs ** 2).mean())
    logger.info("  RMSE       = {:.6f}", np.sqrt((traj_errs ** 2).mean()))
    logger.info("  mean|err|  = {:.6f}",     traj_errs.mean())
    logger.info("  max|err|   = {:.6f}",     traj_errs.max())
    logger.info("  p50/p95/p99 = {:.4f} / {:.4f} / {:.4f}",
                np.quantile(traj_errs, 0.5), np.quantile(traj_errs, 0.95), np.quantile(traj_errs, 0.99))

    # Verdict
    logger.info("")
    if err0.max() < 0.3 and (err0[:, :6].max() < 0.2 and err0[:, 7:13].max() < 0.2):
        logger.info("✓ Policy chunk[0] tracks GT closely — server is healthy.")
    else:
        logger.warning("✗ Outlier max|err| = {:.3f}. Investigate.", err0.max())

    if args.save_path:
        np.savez(args.save_path, chunks=chunks, gt=gts, sample_idxs=sample_idxs, dts=np.array(dts))
        logger.info("Saved trace: {}", args.save_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
