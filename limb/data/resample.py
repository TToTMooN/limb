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

import functools
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# retime_video puts frames on an mp4 track timescale where one frame is a whole
# number of ticks (constant-rate stts), with at least this many ticks per second.
_RETIME_MIN_TIMESCALE = 10_000
# Bitstream filter that rewrites the codec's own frame timing (H.264/HEVC VUI)
# for a remux, and its ticks per frame (H.264 counts field ticks: 2 per frame).
_TIMING_BSF = {"h264": ("h264_metadata", 2), "hevc": ("hevc_metadata", 1)}
# Max allowed |pts_i - i / fps| and |duration_i - 1 / fps| after retiming.
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


def _fraction(value: str) -> Optional[Fraction]:
    """Parse an ffprobe rate such as ``30/1``; None if missing, zero or malformed."""
    try:
        rate = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _probe_video(path: Path, packets: bool = False) -> Optional[dict]:
    """ffprobe the first video stream of ``path`` without decoding.

    Returns ``{"codec", "r_frame_rate", "avg_frame_rate"}`` and, with
    ``packets``, ``"pts"`` (sorted, seconds) and ``"durations"`` (seconds, for
    packets that report one). None if ffprobe is missing or the file can't be read.
    """
    entries = "stream=codec_name,r_frame_rate,avg_frame_rate,time_base" + (":packet=pts,duration" if packets else "")
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries, "-of", "json", str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        probe = json.loads(result.stdout)
        stream = probe["streams"][0]
        info = {
            "codec": stream.get("codec_name"),
            "r_frame_rate": _fraction(stream.get("r_frame_rate")),
            "avg_frame_rate": _fraction(stream.get("avg_frame_rate")),
        }
        if packets:
            time_base = float(Fraction(stream["time_base"]))
            pkts = probe.get("packets", [])
            info["pts"] = np.sort(np.array([p["pts"] for p in pkts if "pts" in p], dtype=np.int64)) * time_base
            info["durations"] = np.array([p["duration"] for p in pkts if "duration" in p], dtype=np.int64) * time_base
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return None
    return info


@functools.lru_cache(maxsize=1)
def _setts_prescale() -> str:
    """``":prescale=1"`` if this ffmpeg's setts filter has the option, else ``""``.

    From FFmpeg 8.1, setts rescales its result from the input to the output
    time base unless ``prescale=1``; retime_video's expressions already give
    output-time-base ticks. Older builds don't know the option.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "bsf=setts"], capture_output=True, text=True, timeout=30, check=False
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    return ":prescale=1" if "prescale" in result.stdout else ""


def _retime_problem(path: Path, rate: Fraction, n_frames: int) -> Optional[str]:
    """Check that ``path`` is a constant-rate ``rate`` video with frame ``i`` at
    ``i / rate``; return what is wrong, or None if it is right.

    Decoders find frames from different fields, so all of them must agree:
    LeRobot's PyAV path uses packet pts, torchcodec's approximate mode uses
    ``r_frame_rate`` and its exact mode uses packet durations. A video with
    fewer than ``n_frames`` frames is checked over the frames it has (and a
    warning is logged): those rows simply have no frame, as before.
    """
    info = _probe_video(path, packets=True)
    if info is None or len(info["pts"]) == 0:
        return "could not read its timestamps"
    if info["r_frame_rate"] != rate:
        return f"r_frame_rate is {info['r_frame_rate']}, expected {rate}"
    avg = info["avg_frame_rate"]
    if avg is None or abs(avg - rate) > rate * 1e-6:
        return f"avg_frame_rate is {avg}, expected {rate}"
    pts = info["pts"]
    if len(pts) < n_frames:
        logger.warning("{} has {} frames for {} steps; trailing steps have no video frame", path, len(pts), n_frames)
    n = min(len(pts), n_frames)
    grid_err = float(np.max(np.abs(pts[:n] - np.arange(n) / float(rate))))
    if grid_err > _RETIME_TOLERANCE_S:
        return f"frame times are off the 1/{rate} grid by up to {grid_err:.2e} s"
    durations = info["durations"]
    if len(durations) and float(np.max(np.abs(durations - 1 / float(rate)))) > _RETIME_TOLERANCE_S:
        return f"packet durations range {durations.min():.6f}-{durations.max():.6f} s, expected {1 / float(rate):.6f}"
    return None


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
    broken = False  # ffmpeg exited early (e.g. no libx264 in this build); its error is on stderr
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            try:
                proc.stdin.write(bgr.tobytes())
            except BrokenPipeError:
                broken = True
                break
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except BrokenPipeError:
            broken = True
        proc.wait()
    if proc.returncode != 0 or broken:
        raise RuntimeError(
            f"ffmpeg re-encode failed for {src_path} (exit {proc.returncode}; see ffmpeg's error above)"
        )


def retime_video(src_path: Path, dst_path: Path, fps: float, n_frames: int) -> str:
    """Write ``src_path`` to ``dst_path`` so frame ``i`` is presented at exactly
    ``i / fps`` seconds, without changing which frames are in it.

    The recorder writes one frame per control tick (60-100 Hz) but encodes the
    mp4 at ``recording_fps`` (30), so in the raw file frame ``i`` sits at
    ``i / 30`` s. A dataset that labels row ``i`` with ``t = i / fps`` must
    retime the video to match, or timestamp-based frame lookup (LeRobot)
    returns frames from the wrong moment.

    Tries a lossless stream-copy remux first (ffmpeg ``setts`` plus the
    codec's ``*_metadata`` bitstream filter, H.264/HEVC only) and falls back to
    a re-encode. Either way the output timing is verified (pts, frame rate
    fields and packet durations, see ``_retime_problem``). ``fps`` is used as a
    fraction with denominator <= 1000, as convert_lerobot computes it.
    Returns ``"remux"`` or ``"reencode"``.
    """
    rate = Fraction(float(fps)).limit_denominator(1000)
    src = _probe_video(src_path)
    src_rate = src["r_frame_rate"] if src else None
    timing_bsf = _TIMING_BSF.get(src["codec"]) if src else None
    if src_rate is not None and timing_bsf is not None:
        # Source pts are k / src_rate (constant-rate recorder output): recover
        # the frame index k and place it at k * step ticks, with one frame
        # exactly `step` ticks so the track is constant-rate at `rate`. Then
        # rewrite the stream's own timing, which still says the recording rate.
        k = -(-_RETIME_MIN_TIMESCALE // rate.numerator)
        timescale, step = rate.numerator * k, rate.denominator * k
        index = f"round((%s-STARTPTS)*TB*{src_rate.numerator}/{src_rate.denominator})"
        bsf_name, ticks_per_frame = timing_bsf
        bsf = (
            f"setts=pts={index % 'PTS'}*{step}:dts={index % 'DTS'}*{step}:duration={step}"
            f":time_base=1/{timescale}{_setts_prescale()},"
            f"{bsf_name}=tick_rate={ticks_per_frame * rate.numerator}/{rate.denominator}"
        )
        cmd = [
            *"ffmpeg -loglevel error -y -i".split(),
            str(src_path),
            *"-map 0:v:0 -c copy -bsf:v".split(),
            bsf,
            *f"-video_track_timescale {timescale}".split(),
            str(dst_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            problem = _retime_problem(dst_path, rate, n_frames)
            if problem is None:
                return "remux"
            logger.warning("Lossless retime of {}: {}; re-encoding instead", src_path, problem)
        else:
            logger.warning("Lossless retime of {} failed ({}); re-encoding instead", src_path, result.stderr.strip())

    _reencode_on_grid(src_path, dst_path, fps)
    problem = _retime_problem(dst_path, rate, n_frames)
    if problem is not None:
        raise RuntimeError(f"could not retime {src_path} to {rate} fps: {problem}")
    return "reencode"
