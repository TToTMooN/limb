"""Resample raw limb episode data to a target fps.

Limb records at the control-loop rate (often 60-100 Hz) but trains policies
at lower rates (typically 30 Hz) so action chunks span a more useful amount
of wall-clock lookahead per step. The previous LeRobot converter just
*relabeled* the recorded frames at a hardcoded 30 fps, leaving the trajectory
slowed-down relative to the original record. This module does honest
resampling: state and action are interpolated onto a regular target time grid,
videos are decimated/duplicated frame-by-frame to match.

Key invariants:
  • Input timestamps.npy is treated as ground truth real time.
  • Target time grid is regular: ``t_i = i / target_fps`` starting at 0.
  • The output number of frames is ``floor(duration * target_fps) + 1``.
  • Continuous signals (joint pos, joint vel, ee_pose, …) use linear interp.
  • Bimodal / threshold-snapped signals (gripper) and discrete signals (video
    frames, camera timestamps) use **zero-order hold** — for each target
    time ``t``, take the most recent source sample whose timestamp is ``<= t``.
    ZOH is causal: never uses a source sample from after the target time, so
    transitions are not blurred or peeked-at-from-the-future.

The same ZOH path works for both downsampling (record fast, train slow) and
upsampling (record slow, train fast); upsampling just holds each source
sample for multiple target frames.

Layout::

    detect_source_fps(timestamps)              → float, Hz
    resample_state_action(...)                 → (target_rel_t, state, action)
    resample_video(src_mp4, dst_mp4, ...)      → writes dst_mp4 at target fps
    retime_video(src_mp4, dst_mp4, fps, n)     → frame i presented at exactly i / fps
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# mp4 track timescale used by retime_video: 1 µs ticks, so presentation times
# are exact to well within LeRobot's default 1e-4 s frame-lookup tolerance.
_RETIME_TIMESCALE = 1_000_000
# Max allowed |pts_i - i / fps| after retiming.
_RETIME_TOLERANCE_S = 1e-5


def detect_source_fps(timestamps: np.ndarray) -> float:
    """Compute the empirical average frame rate of a timestamps.npy.

    Returns 0.0 if there are fewer than 2 timestamps or the recording duration
    is non-positive (corrupt episode).
    """
    if len(timestamps) < 2:
        return 0.0
    dur = float(timestamps[-1] - timestamps[0])
    if dur <= 0:
        return 0.0
    return (len(timestamps) - 1) / dur


def _zoh_indices(src_rel: np.ndarray, tgt_rel: np.ndarray) -> np.ndarray:
    """Zero-order-hold index map: for each ``tgt_rel[i]`` return the largest
    source index ``j`` such that ``src_rel[j] <= tgt_rel[i]``.

    Causal — never points at a source sample from after the target time. When
    ``tgt_rel[i] < src_rel[0]`` (target precedes the first source sample, only
    possible for non-zero start grids which we don't currently use), the
    result is clamped to 0.

    Both arrays must be sorted ascending. Returns a 1-D int64 array.
    """
    idx = np.searchsorted(src_rel, tgt_rel, side="right") - 1
    return np.clip(idx, 0, len(src_rel) - 1).astype(np.int64)


def _interp_columns(src_rel: np.ndarray, src_arr: np.ndarray, tgt_rel: np.ndarray) -> np.ndarray:
    """Linear interpolation along axis 0, column-wise. Output dtype matches
    ``src_arr``; values are computed in float64 then cast.
    """
    out = np.empty((len(tgt_rel), src_arr.shape[1]), dtype=np.float64)
    for d in range(src_arr.shape[1]):
        out[:, d] = np.interp(tgt_rel, src_rel, src_arr[:, d])
    return out.astype(src_arr.dtype)


def _build_target_grid(src_timestamps: np.ndarray, target_fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(src_rel, tgt_rel)`` where both are seconds relative to the
    first source timestamp. ``tgt_rel`` covers ``[0, duration]`` at
    ``target_fps``; samples that would overshoot ``duration`` by more than an
    epsilon are dropped.
    """
    src_rel = src_timestamps - src_timestamps[0]
    duration = float(src_rel[-1])
    n_tgt = int(np.floor(duration * target_fps)) + 1
    tgt_rel = np.arange(n_tgt, dtype=np.float64) / float(target_fps)
    return src_rel.astype(np.float64), tgt_rel[tgt_rel <= duration + 1e-9]


def resample_state_action(
    src_timestamps: np.ndarray,
    src_state: np.ndarray,
    src_action: np.ndarray,
    target_fps: float,
    zoh_action_dims: Tuple[int, ...] = (),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample state and action arrays onto a regular ``target_fps`` grid.

    Parameters
    ----------
    src_timestamps : (N,) float
        Real timestamps (Unix seconds) from the episode's ``timestamps.npy``.
    src_state : (N, state_dim)
        Per-frame state vector (joint_pos + gripper_pos, etc.).
    src_action : (N, action_dim)
        Per-frame action vector.
    target_fps : float
        Output frame rate. Output length is ``floor(duration * target_fps) + 1``.
    zoh_action_dims : tuple of int
        Action dims that are bimodal/binarized (typically gripper). These are
        sampled with zero-order hold (last source value at-or-before the
        target time) so transitions stay crisp and causal instead of
        smearing through intermediate values via linear interp.

    Returns
    -------
    target_rel_t : (N_tgt,) float64
        Target timestamps in seconds, relative to ``src_timestamps[0]``.
    new_state : (N_tgt, state_dim)
        Resampled state (linear interp on all dims).
    new_action : (N_tgt, action_dim)
        Resampled action (linear interp by default; ZOH on dims in
        ``zoh_action_dims``).
    """
    assert src_state.shape[0] == src_action.shape[0] == len(src_timestamps)
    src_rel, tgt_rel = _build_target_grid(src_timestamps, target_fps)
    if len(tgt_rel) == 0:
        return np.zeros(0, dtype=np.float64), src_state[:0], src_action[:0]

    new_state = _interp_columns(src_rel, src_state, tgt_rel)
    new_action = _interp_columns(src_rel, src_action, tgt_rel)
    if zoh_action_dims:
        idx = _zoh_indices(src_rel, tgt_rel)
        for d in zoh_action_dims:
            if 0 <= d < new_action.shape[1]:
                new_action[:, d] = src_action[idx, d]
    return tgt_rel, new_state, new_action


def resample_video(
    src_path: Path,
    dst_path: Path,
    src_timestamps: np.ndarray,
    target_rel_t: np.ndarray,
    target_fps: float,
    codec: str = "auto",
) -> int:
    """Resample a recorded mp4 to ``target_fps`` using zero-order hold.

    For each target time, pick the most recent source frame at-or-before that
    time and emit it. Both ``src_timestamps`` and ``target_rel_t`` must be
    sorted ascending. The mp4 is written via ``robocam.AsyncVideoWriter``
    (same NVENC/H.265 path as the recorder).

    Returns the number of frames written.
    """
    from robocam import AsyncVideoWriter

    src_rel = src_timestamps - src_timestamps[0]
    needed = _zoh_indices(src_rel, target_rel_t)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open source video: {src_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = AsyncVideoWriter(path=str(dst_path), width=w, height=h, fps=round(target_fps))
    writer.start()

    # `needed` is monotonic non-decreasing because both inputs are sorted; this
    # lets us walk the source video forward only — each frame is decoded at
    # most once and held across however many target frames it backs.
    cur_src_idx = -1
    cur_frame_rgb = None
    blank = None
    for tgt_src_idx in needed:
        while cur_src_idx < int(tgt_src_idx):
            ok, bgr = cap.read()
            if not ok:
                break
            cur_src_idx += 1
            cur_frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if cur_frame_rgb is None:
            # Source EOF before we reached the needed index. Shouldn't happen
            # for healthy episodes (timestamps.npy and the mp4 frame count
            # match by construction in the recorder), but emit blanks rather
            # than crash so the user can still inspect the dataset.
            if blank is None:
                blank = np.zeros((h, w, 3), dtype=np.uint8)
                logger.warning("Source video ended before target idx — writing blank frames")
            writer.write(blank)
        else:
            writer.write(cur_frame_rgb)

    cap.release()
    writer.stop()
    return len(needed)


def _probe_frame_rate(path: Path) -> Optional[Fraction]:
    """Return the nominal frame rate of the first video stream, or None."""
    cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0".split()
    try:
        result = subprocess.run([*cmd, str(path)], capture_output=True, text=True, timeout=30, check=False)
        rate = Fraction(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _probe_presentation_times(path: Path) -> Optional[np.ndarray]:
    """Return the sorted packet presentation times (seconds) of the first video
    stream, read from the container without decoding. None on failure."""
    cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=time_base:packet=pts -of json".split()
    try:
        result = subprocess.run([*cmd, str(path)], capture_output=True, text=True, timeout=120, check=False)
        probe = json.loads(result.stdout)
        time_base = Fraction(probe["streams"][0]["time_base"])
        pts = np.array([p["pts"] for p in probe["packets"] if "pts" in p], dtype=np.int64)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return None
    return np.sort(pts) * float(time_base)


def _max_grid_error(path: Path, fps: float, n_frames: int) -> Optional[float]:
    """Max |pts_i - i / fps| over the first ``n_frames`` frames of ``path``.

    None if the timestamps can't be read. A video with fewer than ``n_frames``
    frames is checked over the frames it has (and a warning is logged): those
    rows simply have no frame, as before.
    """
    pts = _probe_presentation_times(path)
    if pts is None or len(pts) == 0:
        return None
    if len(pts) < n_frames:
        logger.warning("{} has {} frames for {} steps; trailing steps have no video frame", path, len(pts), n_frames)
    n = min(len(pts), n_frames)
    return float(np.max(np.abs(pts[:n] - np.arange(n) / fps)))


def _reencode_on_grid(src_path: Path, dst_path: Path, fps: float) -> None:
    """Re-encode ``src_path`` frame-by-frame with a constant ``fps`` timeline.

    Fallback for ffmpeg builds whose ``setts`` bitstream filter can't do the
    lossless retime (it needs the ``time_base`` option). ``fps`` is passed to
    ffmpeg as a fraction with denominator <= 1000, so it is exact only for
    rates of that form (see convert_lerobot).
    """
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open source video: {src_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rate = Fraction(float(fps)).limit_denominator(1000)
    cmd = [
        *"ffmpeg -loglevel error -y -f rawvideo -pix_fmt bgr24 -s".split(),
        f"{w}x{h}",
        "-r",
        f"{rate.numerator}/{rate.denominator}",
        *"-i - -an -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p".split(),
        str(dst_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            proc.stdin.write(bgr.tobytes())
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg re-encode failed for {src_path} (exit {proc.returncode})")


def retime_video(src_path: Path, dst_path: Path, fps: float, n_frames: int) -> str:
    """Write ``src_path`` to ``dst_path`` so frame ``i`` is presented at exactly
    ``i / fps`` seconds, without changing which frames are in it.

    The recorder writes one frame per control tick (60-100 Hz) but encodes the
    mp4 at ``recording_fps`` (30), so in the raw file frame ``i`` sits at
    ``i / 30`` s. A dataset that labels row ``i`` with ``t = i / fps`` must
    retime the video to match, or timestamp-based frame lookup (LeRobot)
    returns frames from the wrong moment.

    Tries a lossless stream-copy remux first (ffmpeg ``setts`` bitstream
    filter) and falls back to a re-encode. Either way the output timestamps are
    verified. Returns ``"remux"`` or ``"reencode"``.
    """
    src_rate = _probe_frame_rate(src_path)
    if src_rate is not None:
        # Source pts are k / src_rate (constant-rate recorder output): recover
        # the frame index k, then place it at k / fps on a 1 µs time base.
        index = f"round((%s-STARTPTS)*TB*{src_rate.numerator}/{src_rate.denominator})"
        ticks = f"*{_RETIME_TIMESCALE}/{float(fps)!r}"
        setts = (
            f"setts=pts={index % 'PTS'}{ticks}:dts={index % 'DTS'}{ticks}"
            f":duration=round({_RETIME_TIMESCALE}/{float(fps)!r}):time_base=1/{_RETIME_TIMESCALE}"
        )
        cmd = [
            *"ffmpeg -loglevel error -y -i".split(),
            str(src_path),
            *"-map 0:v:0 -c copy -bsf:v".split(),
            setts,
            *f"-video_track_timescale {_RETIME_TIMESCALE}".split(),
            str(dst_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            err = _max_grid_error(dst_path, fps, n_frames)
            if err is not None and err <= _RETIME_TOLERANCE_S:
                return "remux"
            logger.warning("Lossless retime of {} is off by {} s; re-encoding instead", src_path, err)
        else:
            logger.warning("Lossless retime of {} failed ({}); re-encoding instead", src_path, result.stderr.strip())

    _reencode_on_grid(src_path, dst_path, fps)
    err = _max_grid_error(dst_path, fps, n_frames)
    if err is None or err > _RETIME_TOLERANCE_S:
        raise RuntimeError(f"could not retime {src_path} to {fps:.4f} fps (max timestamp error: {err} s)")
    return "reencode"
