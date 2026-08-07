"""Image-grounded vial perception for the sub-task loop.

Backend switch (SUBRL_PERCEPTION_BACKEND or ctor arg):
  - "gemini_er"   : Gemini-Robotics-ER via the LiteLLM API (now; no local GPU).
  - "sam3_owlvit" : OWLv2 /detect (+ SAM3 /segment) HTTP servers on the H200 (later).

NON-BLOCKING (review C4): all network calls run on a single background worker thread.
`detect_vial()` never blocks the 30 Hz control loop — it returns the cached result for
that camera immediately and, if the per-camera throttle window has elapsed, enqueues a
refresh with the latest frame. The throttle timestamp is armed when a request is
ISSUED (not on success), so a down endpoint retries once per window, not every tick.

Per-camera cache (review M19): results are keyed by a caller-supplied ``cam`` tag so a
multi-view sweep (wrist + head) gets genuinely independent verdicts.

Fail-open (review L27): only successful parses are cached. HTTP errors raise via
``raise_for_status`` inside the worker and leave the cache at the last good value
(or None = "detector unavailable"), never a bogus ``found: False``.

Provides the two primitives the authored artifacts expect:
  detect_vial(rgb, cam="default") -> {"found", "point":[x,y], "bbox":[...], "area"} | None
  vial_visible(obs, side) -> bool   (True when the detector is unavailable — proprio decides)
"""
from __future__ import annotations

import base64
import io
import os
import pathlib
import queue
import re
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

_LITELLM_URL = "https://litellm.avantrobotics.ai/v1/chat/completions"
_ER_MODEL = "gemini-robotics-er-1.6-preview"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_HELD_PROMPT = (
    "This is the robot's RIGHT WRIST camera. Question: is a glass VIAL (small test "
    "tube with a black cap) currently HELD — clenched BETWEEN the robot gripper "
    "fingers, grasped and lifted off the table? A vial merely NEAR the gripper or "
    "standing on the table below does NOT count as held. Respond ONLY with JSON: "
    "{\"held\": true} or {\"held\": false}."
)
_RELEASE_PROMPT = (
    "This is the robot's RIGHT WRIST camera. The robot gripper is holding a glass "
    "vial (small test tube with a black cap) and is lowering it to set it down "
    "UPRIGHT on the table. Judge the CURRENT frame: (1) \"upright\": is the vial "
    "VERTICAL (aligned with gravity, not tilted)? (2) \"near_table\": is the vial's "
    "BOTTOM touching the table surface or within about 1 cm of it? Respond ONLY "
    "with JSON: {\"upright\": true, \"near_table\": true} (booleans)."
)
_GPT_MODEL = "gpt-5.5"        # advanced-VLM option (backend="gpt"): stronger scene
                              # understanding (stand-vs-table, fallen vials) at higher
                              # per-call latency than Gemini-ER's pointing model
_POINT_PROMPT = (
    "Detect every glass VIAL (small test tube with a black cap) that is ON THE TABLE "
    "SURFACE itself, and classify its pose. Do not skip fallen or tipped-over vials — "
    "report them too. STRICTLY EXCLUDE: (a) the white RACK/STAND on the table — its "
    "black circular holes are NOT vials, do not report the rack or its holes; (b) "
    "vials inserted in that rack; (c) vials held in a robot gripper. Report ONLY "
    "vials resting on the bare table surface. For each vial set \"pose\" to "
    "\"standing\" (upright, cap on top) or \"fallen\" (tipped over, lying on its "
    "side, or rolling). IMPORTANT viewpoint rule: this may be a wrist camera looking "
    "STRAIGHT DOWN at the table — from above, a STANDING vial appears as just a "
    "SMALL DARK CIRCLE (its black cap) with little or no tube visible; that IS a "
    "standing vial, report it with pose \"standing\". A FALLEN vial seen from above "
    "shows its full elongated tube lying on the table. Respond ONLY with JSON: "
    "[{\"box_2d\":[ymin,xmin,ymax,xmax], \"pose\":\"standing\"}] with box_2d "
    "normalized to 0-1000 (origin top-left). If no vial is on the table surface, "
    "respond with []."
)


_QWEN_URL = "https://router.huggingface.co/v1/chat/completions"

# CONFIRM-lane classifier prompt (user 2026-07-28): at the selector's commit moment
# the wrist camera looks STRAIGHT DOWN — a standing vial is just a black cap circle,
# and "is there a standing vial?" fails on that view (Gemini-ER 3/6, wrong BLOCKS on
# true entries = the "VLA never switches" sessions). This viewpoint-aware 4-way
# classification measured on 2026-07-28 GT frames: Qwen3.5-27B (no-think) 5/5 @
# 0.9 s median; gemini-3.5-flash 6/6 behaviorally @ 2.6 s; gemini-robotics-er 3/6.
_TOPDOWN_PROMPT = (
    "You are looking through a robot's RIGHT WRIST camera pointing STRAIGHT DOWN "
    "at a wooden table, between the two dark gripper fingers (bottom corners). "
    "Classify what is between/below the fingers:\n"
    "- \"standing_vial\": a small glass vial standing UPRIGHT on the wood table — "
    "seen from above it appears as a small dark CAP CIRCLE (coin-sized dot), or a "
    "short vial seen at a slight angle.\n"
    "- \"fallen_vial\": a vial LYING on its side on the table — an elongated tube shape.\n"
    "- \"stand\": the vial stand/rack — a metal or white plate with a grid of "
    "circular slots (empty or holding inserted tubes).\n"
    "- \"none\": bare table or anything else.\n"
    "Respond ONLY with JSON: {\"target\": \"...\"}"
)


# INSERTION check (insert sub-task, 2026-07-31): single-frame classification measured
# INFEASIBLE (best cell 82%, most at/below the always-inserted baseline — occlusion +
# black-cap-vs-empty-hole ambiguity). The working design is a BEFORE/AFTER visual diff
# on the SAME camera: 16/16 with gemini-robotics-er-1.6-preview on demo GT pairs
# (4.2 s p50 — fine for an event-conditioned per-episode terminal reward).
_INSERT_MODEL = os.environ.get("SUBRL_INSERT_MODEL", "gemini-robotics-er-1.6-preview")
_INSERT_PAIR_PROMPT = (
    "These are two frames from the SAME robot camera: image 1 is BEFORE, image 2 is "
    "AFTER a robot attempted to insert one vial into the white vial stand (white rack "
    "with round holes, on a wooden table). Several holes may already have contained "
    "vials BEFORE — ignore those. Compare the two images and decide whether the "
    "insertion completed between them:\n"
    "- \"success\": in image 2 there is ONE MORE vial fully seated in a stand hole "
    "than in image 1 (a hole that was empty in image 1 now holds an upright vial), "
    "and the gripper has released it.\n"
    "- \"not_done\": no new vial appeared in any hole (the vial is still held in the "
    "air, still gripped, or nothing changed).\n"
    "- \"failed\": a vial ended up tipped over / on the table / lying across the stand.\n"
    "- \"cannot_tell\": the stand is not visible well enough in one of the images.\n"
    "STRICT JSON only: {\"result\": \"<success|not_done|failed|cannot_tell>\"}"
)


def _load_hf_token() -> Optional[str]:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    p = pathlib.Path.home() / ".cache/huggingface/token"
    return p.read_text().strip() if p.exists() else None


def _load_key() -> Optional[str]:
    if os.environ.get("SUBRL_LLM_KEY"):
        return os.environ["SUBRL_LLM_KEY"]
    for p in (pathlib.Path.home() / ".subrl_llmkey",
              pathlib.Path("/home/ssc/Desktop/research/cap-x/.llmkey")):
        if p.exists():
            return p.read_text().strip()
    return None


def _png_data_url(rgb: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _cv2_png_data_url(rgb: np.ndarray) -> str:
    """Lossless PNG via cv2 (GIL-releasing, ~8 ms @448 px vs PIL's 20 ms GIL-held).
    Used for the SAM3 payload: its confidence scores are calibrated on clean pixels —
    JPEG q90 HALVED the fallen-tube score (0.43 -> 0.21, under the 0.25 threshold)."""
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        return _png_data_url(rgb)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _jpeg_data_url(rgb: np.ndarray) -> str:
    """JPEG via cv2: releases the GIL and encodes 3-5x faster than PIL PNG. PIL PNG
    encodes of 448-px frames in the worker threads starved the 30 Hz act() thread
    (2026-07-07 23:15 run: agent.act spiked to 170-300 ms, loop at 12-17 Hz)."""
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return _png_data_url(rgb)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class VialDetector:
    def __init__(self, backend: str | None = None, throttle_s: float = 1.0,
                 owlvit_url: str | None = None, sam3_url: str | None = None,
                 min_area_px: float = 20.0, max_tokens: int = 1024,
                 request_timeout_s: float = 10.0, model: str | None = None,
                 workers: int | None = None, release_backend: str | None = None,
                 audit_dir: str | None = None,
                 lane_throttles: dict | None = None,
                 send_depth: bool = False,
                 depth_scales: dict | None = None,
                 intrinsics: dict | None = None):
        self.backend = backend or os.environ.get("SUBRL_PERCEPTION_BACKEND", "gemini_er")
        # DETECTION AUDIT (user request 2026-07-07): save every detection-lane verdict
        # + its exact input frame so runs can be graded offline (was the fallen vial
        # really fallen?). Auto-on for the sam3 backend; ~150 KB/frame at the 2 s
        # per-camera throttle, frame saving capped at _AUDIT_MAX_FRAMES per session.
        self.audit_dir = (audit_dir or os.environ.get("SUBRL_PERCEPTION_AUDIT")
                          or ("logs/sam3_audit" if self.backend == "sam3" else None))
        self._audit_n = 0
        self.model = model                     # backend-specific override (gpt: default gpt-5.5)
        # HYBRID split (user architecture 2026-07-07): SAM3 is a DETECTOR — presence,
        # boxes/masks, geometric standing/fallen — but cannot answer SEMANTIC queries
        # (check_release: "is the HELD vial upright with its bottom AT the table?").
        # When the main backend is a pure detector, semantic queries route to this VLM
        # backend instead (auto: GPT-5.5 if an OpenAI key exists, else Gemini-ER).
        self.release_backend = release_backend
        # Worker threads: remote-VLM calls are SLOW (measured 2026-07-07: Gemini-ER via
        # LiteLLM ~5 s, GPT-5.5 ~8-11 s) — one worker serializes the 3 cameras into a
        # 15-30 s per-camera refresh cycle; 3 workers keep staleness ≈ one call. The
        # threads block on network (GIL-cheap; the 320-512 px PNG encode is ~ms — the
        # 2026-07-06 GIL starvation came from duplicate detectors at full 640 px).
        if workers is None:
            # all remote backends get 3 workers: the queue also carries the SEMANTIC
            # (gemini) release/held calls — one worker would serialize them behind
            # detection and stale both lanes. Local sam3_owlvit stays single.
            workers = 1 if self.backend == "sam3_owlvit" else 3
        self._n_workers = max(1, int(workers))
        self.throttle_s = float(throttle_s)
        # PER-LANE throttle overrides (2026-08-04, SAM3 saturation fix): keys are
        # issue-key prefixes, e.g. {"head_camera": 8.0, "stand:": 4.0} — the DECISIVE
        # wrist-detect lane keeps the base cadence while secondary lanes refresh
        # slower, so the shared SAM3 server can actually keep the wrist verdict fresh.
        self.lane_throttles = dict(lane_throttles or {})
        self.owlvit_url = owlvit_url or os.environ.get("SUBRL_OWLVIT_URL", "http://127.0.0.1:8117")
        self.sam3_url = sam3_url or os.environ.get("SUBRL_SAM3_URL", "http://127.0.0.1:8114")
        # DEPTH MODE (user 2026-08-05, lingbot_depth only): the agent stashes each
        # camera's aligned depth frame here via observe_depth(); _lingbot ships it
        # (uint16 PNG) + the probed intrinsics with the detect request, turning the
        # RGB-only depth-completion guess into sensor-conditioned METRIC geometry.
        # depth_scales: cam -> meters-per-unit. PER-CAMERA because the rig mixes
        # models (probed 2026-08-05: D405 wrists = 1e-4, D455 head = 1e-3).
        # intrinsics: cam -> [fx, fy, cx, cy] at the 640x480 color stream; _lingbot
        # rescales to the frame it actually sends. Intrinsics ride ONLY with depth:
        # RGB-only requests keep the service's fx=615 default that the classifier
        # gates were benchmarked on (6/6 GT) — don't silently retune that path.
        self.send_depth = bool(send_depth)
        self.depth_scales = dict(depth_scales or {})
        self.intrinsics = dict(intrinsics or {})
        self._depth: Dict[str, np.ndarray] = {}   # cam -> last depth frame (HD ticks)
        self.min_area_px = float(min_area_px)
        self.max_tokens = int(max_tokens)
        self.request_timeout_s = float(request_timeout_s)
        self._key = _load_key()
        self._lock = threading.Lock()
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}   # cam -> last GOOD result (or None)
        self._t_issued: Dict[str, float] = {}                    # cam -> last request-ISSUE time
        self._inflight: set[str] = set()
        self._q: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue(maxsize=8)
        self._threads = [threading.Thread(target=self._work, daemon=True)
                         for _ in range(self._n_workers)]
        for t in self._threads:
            t.start()

    # ---- primitives the authored artifacts call -------------------------
    def observe_depth(self, obs) -> None:
        """Per-tick agent hook (depth mode): stash the latest aligned depth frame
        per camera. Depth arrives only on the HD ticks (launch forwards it with
        rgb_hd — Portal budget), so the stash IS the cache between arrivals, and
        it stays paired with the agent's rgb_hd cache from the same tick."""
        if not self.send_depth:
            return
        try:
            now = time.monotonic()
            for cam, entry in obs.items():
                if not isinstance(entry, dict):
                    continue
                imgs = entry.get("images")
                if isinstance(imgs, dict) and imgs.get("depth") is not None:
                    with self._lock:
                        # (frame, stamp): _issue drops frames older than ~2 s so a
                        # depth-stream dropout degrades to the benchmarked RGB-only
                        # request instead of silently pairing live rgb with a frozen
                        # stale depth map (review 2026-08-05).
                        self._depth[cam] = (np.asarray(imgs["depth"]), now)
        except Exception:
            pass                                    # fail-open: RGB-only request

    def detect_vial(self, rgb, cam: str = "default") -> Optional[Dict[str, Any]]:
        """Immediate, non-blocking: return the cached result for `cam` and (if the
        throttle window elapsed) enqueue a background refresh with this frame.
        Returns None while no successful detection has ever completed for `cam`."""
        return self._issue(cam, rgb)

    def check_release(self, rgb, cam: str = "right_wrist_camera") -> Optional[Dict[str, Any]]:
        """Wrist-camera RELEASE gate (user rule 2026-07-07): before the reset opens
        the gripper, the wrist image must confirm the HELD vial is upright with its
        bottom at the table. Non-blocking/cached like detect_vial; returns
        {'upright': bool, 'near_table': bool} or None while unavailable."""
        return self._issue(f"release:{cam}", rgb)

    def check_held(self, rgb, cam: str = "right_wrist_camera") -> Optional[Dict[str, Any]]:
        """Wrist-camera SUCCESS confirmation (user rule 2026-07-07): the verifier's
        grasp+lift success must also be image-confirmed — the vial CLENCHED between
        the gripper fingers (CaP-X insight: proximity is not grasp). Returns
        {'held': bool} or None while unavailable."""
        return self._issue(f"held:{cam}", rgb)

    def check_stand(self, rgb, cam: str = "right_wrist_camera") -> Optional[Dict[str, Any]]:
        """INSERT-entry confirm (user 2026-07-31: the wrist camera must confirm the
        gripper is ABOVE THE STAND before the insert skill starts — short-horizon
        rollouts). LOCAL SAM3 lane, no VLM: prompt-segments the stand/rack in the
        wrist view; the rack's hole grid also fires the cap prompt (measured: 12-22
        boxes when staring at the rack), used as a second cue. Non-blocking/cached;
        returns {'stand_visible': bool, ...} or None while unavailable."""
        return self._issue(f"stand:{cam}", rgb)

    _SAM3_STAND_PROMPT = "white vial rack with a grid of round holes"

    def _stand(self, rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """SAM3 stand-visibility verdict for the wrist view (entry confirm lane).
        stand_visible = the rack SEGMENT covers >= 2% of the view (lowered from 4%,
        live session 204341 2026-08-03: at the true hover moment the gripper + held
        vial OCCLUDE the rack — approach glimpses ran 0.031-0.036 and the old 0.04
        never fired; the rack prompt is stand-specific so table views cannot false-
        fire it) OR the cap prompt fires on the hole grid (>= 6 boxes, kept strict:
        a 4-vial table scene shows 1-4 caps and must NOT pass). Occlusion at the
        hover itself is bridged by the SELECTOR's latch, not by weaker thresholds."""
        # RACK-SEGMENT ONLY since 2026-08-04 (GT diagnosis, session 213025): the
        # second cap-prompt call doubled this lane's SAM3 cost and the shared server
        # saturated — wrist-detect verdicts went 5-15 s stale and the grasp entries
        # fired late or never. The rack prompt alone carried every measured confirm.
        segs = self._sam3_seg(rgb, self._SAM3_STAND_PROMPT)
        rack_area = max((a for _b, s, _e, a in segs if s >= 0.2), default=0.0)
        return {"stand_visible": bool(rack_area >= 0.02), "rack_area_frac": float(rack_area),
                "cap_boxes": 0, "backend": "sam3_stand"}

    def check_inserted(self, before_rgb, after_rgb, cam: str = "left_wrist_camera",
                       epoch: int = 0) -> Optional[Dict[str, Any]]:
        """INSERT sub-task terminal check (2026-07-31): BEFORE/AFTER visual diff on
        one camera — did a NEW vial appear seated in a stand hole between the two
        frames? Non-blocking/cached like the other lanes; `epoch` (bump per episode)
        keys the cache so a previous episode's verdict can never leak into the next.
        Returns {'result': 'success'|'not_done'|'failed'|'cannot_tell'} or None while
        the verdict is pending."""
        pair = np.stack([np.asarray(before_rgb), np.asarray(after_rgb)])
        return self._issue(f"inserted:{cam}:{int(epoch)}", pair)

    def _issue(self, key: str, rgb) -> Optional[Dict[str, Any]]:
        try:
            thr = self.throttle_s
            for prefix, t in self.lane_throttles.items():
                if key.startswith(prefix):
                    thr = float(t)
                    break
            now = time.monotonic()
            with self._lock:
                due = (now - self._t_issued.get(key, -1e9)) >= thr
                busy = key in self._inflight
                if due and not busy:
                    self._t_issued[key] = now          # arm throttle at ISSUE time (C4)
                    self._inflight.add(key)
                    # Depth snapshot AT ISSUE TIME (review 2026-08-05): the worker may
                    # dequeue seconds later (ER/SAM3 calls occupy the pool) and the
                    # stash refreshes every HD tick — reading it at dequeue time paired
                    # the queued rgb with a depth frame from a camera that had moved.
                    # Detect-lane keys ARE camera names; prefixed lanes snapshot None.
                    dep = None
                    if self.send_depth and self.backend == "lingbot_depth":
                        ent = self._depth.get(key)
                        if ent is not None and (now - ent[1]) <= 2.0:
                            dep = ent[0]
                    try:
                        self._q.put_nowait((key, np.asarray(rgb), dep))
                    except queue.Full:
                        self._inflight.discard(key)
                return self._cache.get(key)
        except Exception:
            return None                                # fail-open: proprio decides

    def vial_visible(self, obs, side: str = "right") -> bool:
        """True if any consulted camera sees a vial; True if the detector is entirely
        unavailable (fail-open); False only when detection WORKED and found nothing
        on every consulted view (review M20).

        ALWAYS consults ALL THREE cameras (per user rule): the wrist cam only sees where
        the gripper points and the head cam alone misses occluded/edge placements —
        single-view presence checks give false 'vial lost' verdicts."""
        any_result = False
        for key in (f"{side}_wrist_camera", "head_camera",
                    f"{'left' if side == 'right' else 'right'}_wrist_camera"):
            try:
                rgb = obs[key]["images"].get("rgb_hd", obs[key]["images"]["rgb"])
            except Exception:
                continue
            r = self.detect_vial(rgb, cam=key)
            if r is None:
                continue                                # this view unavailable; try the next
            any_result = True
            if r.get("found"):
                return True
        return not any_result                           # no working view -> fail-open True

    # ---- background worker ----------------------------------------------
    def _work(self) -> None:
        while True:
            key, rgb, dep = self._q.get()
            try:
                result = (self._release(rgb) if key.startswith("release:")
                          else self._held(rgb) if key.startswith("held:")
                          else self._inserted_pair(rgb) if key.startswith("inserted:")
                          else self._stand(rgb) if key.startswith("stand:")
                          else self._detect(rgb, key, dep))
                if result is not None:                  # cache only good parses (L27)
                    with self._lock:
                        self._cache[key] = result
                    self._audit(key, rgb, result)
            except Exception:
                pass                                    # keep last good cache; throttle already armed
            finally:
                with self._lock:
                    self._inflight.discard(key)

    _AUDIT_MAX_FRAMES = 2000

    def _audit(self, cam: str, rgb: np.ndarray, result: Dict[str, Any]) -> None:
        """Append the verdict to <audit_dir>/audit.jsonl and save the input frame
        (until the cap) — runs in the worker thread, never blocks the control loop."""
        if not self.audit_dir or cam.startswith(("release:", "held:")):
            return
        try:
            import json as _json
            import time as _time
            d = pathlib.Path(self.audit_dir)
            d.mkdir(parents=True, exist_ok=True)
            ts = _time.time()
            fname = None
            with self._lock:
                self._audit_n += 1
                n = self._audit_n
            if n <= self._AUDIT_MAX_FRAMES:
                import cv2
                fname = f"{ts:.2f}_{cam}.jpg"
                cv2.imwrite(str(d / fname),
                            cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            rec = {"t": round(ts, 2), "cam": cam, "frame": fname,
                   **{k: result.get(k) for k in
                      ("found", "standing_count", "fallen_count", "count", "boxes")}}
            with self._lock:
                with (d / "audit.jsonl").open("a") as f:
                    f.write(_json.dumps(rec) + "\n")
        except Exception:
            pass

    # ---- backends ---------------------------------------------------------
    def _detect(self, rgb: np.ndarray, cam: str = "default",
                depth: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        if self.backend == "sam3":
            return self._sam3(rgb)
        if self.backend == "sam3_owlvit":
            return self._owlvit(rgb)
        if self.backend in ("gpt", "openai"):
            return self._gpt(rgb)
        if self.backend == "topdown":
            return self._topdown(rgb)
        if self.backend == "lingbot_depth":
            return self._lingbot(rgb, cam, depth)   # depth = issue-time snapshot
        return self._gemini(rgb)

    _LINGBOT_URL = os.environ.get("SUBRL_LINGBOT_URL", "http://127.0.0.1:8115")

    _INTR_REF = (640, 480)      # resolution the config intrinsics were probed at

    def _lingbot(self, rgb: np.ndarray, cam: str = "default",
                 depth: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """lingbot-depth detection lane (2026-08-04): metric-depth completion ->
        raised-blob geometry -> vials with TRUE-3D standing/fallen (the transparent
        tube resolves in depth where SAM3's mask only ever catches the cap; 5/6->
        6/6 on GT, GPU 112 ms). Serve: services/serve_lingbot_depth.py (:8115).
        DEPTH MODE (2026-08-05): `depth` is the ISSUE-TIME snapshot of the
        camera's aligned depth (paired with this rgb when both were enqueued —
        never re-read at dequeue time, review fix). It rides along (uint16 PNG —
        lossless, like the RGB) with the per-camera depth_scale and the probed
        intrinsics rescaled to the sent frame, conditioning the completion on
        real sensor depth. No depth -> the exact benchmarked RGB-only request
        (no intrinsics override either).
        Returns the standard detector dict (found/standing_count/bbox/pose) plus
        fallen_count/detections and depth_sent."""
        import base64 as _b64

        import cv2
        import requests
        # PNG (lossless): JPEG-90 artifacts measurably flipped the monocular-depth
        # blob geometry (2026-08-04: fallen verdicts turned "standing" through the
        # jpeg round-trip while identical PNG frames classified correctly).
        ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(rgb, np.uint8),
                                                    cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        payload: Dict[str, Any] = {"image_base64": _b64.b64encode(buf).decode()}
        if self.send_depth:
            if depth is not None:
                H, W = rgb.shape[:2]
                if depth.shape[:2] != (H, W):       # NEAREST keeps uint16 values valid
                    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
                okd, dbuf = cv2.imencode(".png", np.asarray(depth, np.uint16))
                if okd:
                    payload["depth_base64"] = _b64.b64encode(dbuf).decode()
                    payload["depth_scale"] = float(self.depth_scales.get(cam, 0.001))
                    intr = self.intrinsics.get(cam)
                    if intr is not None:
                        sx = W / float(self._INTR_REF[0])
                        sy = H / float(self._INTR_REF[1])
                        payload.update(fx=float(intr[0]) * sx, fy=float(intr[1]) * sy,
                                       cx=float(intr[2]) * sx, cy=float(intr[3]) * sy)
        r = requests.post(f"{self._LINGBOT_URL.rstrip('/')}/detect",
                          json=payload, timeout=self.request_timeout_s)
        r.raise_for_status()
        d = r.json()
        if "found" not in d:
            return None
        d["depth_sent"] = "depth_base64" in payload
        return d

    def _inserted_pair(self, pair: np.ndarray) -> Optional[Dict[str, Any]]:
        """BEFORE/AFTER two-image insertion diff (see _INSERT_PAIR_PROMPT). `pair` is
        np.stack([before, after]). Runs on the lab LiteLLM proxy; model from
        SUBRL_INSERT_MODEL (default gemini-robotics-er-1.6-preview — 16/16 on demo GT
        pairs at 640 px; gemini-3.5-flash is the 13/16 fallback)."""
        import re as _re

        import requests
        if not self._key or pair.ndim != 4 or pair.shape[0] != 2:
            return None
        content = [{"type": "text", "text": "Image 1 (BEFORE):"},
                   {"type": "image_url",
                    "image_url": {"url": _jpeg_data_url(self._downscale(pair[0], 640))}},
                   {"type": "text", "text": "Image 2 (AFTER):"},
                   {"type": "image_url",
                    "image_url": {"url": _jpeg_data_url(self._downscale(pair[1], 640))}},
                   {"type": "text", "text": _INSERT_PAIR_PROMPT}]
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # max_tokens 2000 (replay verification 2026-07-31, CRITICAL #3): ER-1.6 spends
        # ~190 tokens as REASONING before the answer — at 200 the content truncated
        # mid-JSON, the parse failed, and every episode silently exited through the
        # lenient proprio path. The groundwork 16/16 measurement used 2000.
        r = requests.post(_LITELLM_URL, timeout=self.request_timeout_s, verify=False,
                          headers={"Authorization": f"Bearer {self._key}"},
                          json={"model": _INSERT_MODEL, "temperature": 0.0, "max_tokens": 2000,
                                "messages": [{"role": "user", "content": content}]})
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"] or ""
        m = _re.search(r'"result"\s*:\s*"(\w+)"', txt)
        if m is None or m.group(1) not in ("success", "not_done", "failed", "cannot_tell"):
            return None
        return {"result": m.group(1)}

    def _topdown(self, rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """CONFIRM-lane backend (user 2026-07-28): viewpoint-aware 4-way wrist-view
        classification (standing_vial / fallen_vial / stand / none) instead of open
        detection — see _TOPDOWN_PROMPT. `model` routes the call: "Qwen/..." -> the
        HF Inference-Providers router (needs funded HF credits; thinking disabled),
        anything else -> the lab LiteLLM proxy (default gemini-3.5-flash).
        Only "standing_vial" maps to found=True — fallen vials, the stand, and bare
        table all BLOCK the selector's VLA->RL entry."""
        import re as _re

        import requests
        model = self.model or "gemini-3.5-flash"
        content = [{"type": "text", "text": _TOPDOWN_PROMPT},
                   {"type": "image_url", "image_url": {"url": _jpeg_data_url(self._downscale(rgb))}}]
        if "/" in model:                                   # HF router (Qwen/...)
            tok = _load_hf_token()
            if not tok:
                return None
            r = requests.post(_QWEN_URL, timeout=self.request_timeout_s,
                              headers={"Authorization": f"Bearer {tok}"},
                              json={"model": model, "temperature": 0.0, "max_tokens": 200,
                                    "chat_template_kwargs": {"enable_thinking": False},
                                    "messages": [{"role": "user", "content": content}]})
        else:                                              # LiteLLM proxy (gemini-*)
            if not self._key:
                return None
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            r = requests.post(_LITELLM_URL, timeout=self.request_timeout_s, verify=False,
                              headers={"Authorization": f"Bearer {self._key}"},
                              json={"model": model, "temperature": 0.0, "max_tokens": 500,
                                    "messages": [{"role": "user", "content": content}]})
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"] or ""
        m = _re.search(r'"target"\s*:\s*"(\w+)"', txt)
        if m is None:
            return None
        target = m.group(1)
        standing = target == "standing_vial"
        return {"found": standing, "standing_count": 1 if standing else 0,
                "pose": "standing" if standing else ("fallen" if target == "fallen_vial" else None),
                "target": target, "backend": "topdown", "model": model}

    # Box elongation threshold for the GEOMETRIC pose rule (sam3 backend): a STANDING
    # vial seen from the (top-down) wrist camera is a compact blob — the black cap —
    # while a FALLEN vial's tube is elongated. Unlike the VLM pose judgment (which
    # misread top-down caps, 18:49 run), this is deterministic geometry.
    FALLEN_ELONG = 1.8

    def _boxes_to_pose_result(self, boxes, elongations=None) -> Dict[str, Any]:
        """Instance boxes [x1,y1,x2,y2] (+ optional per-instance MASK elongations) ->
        the standard detector result dict with geometric standing/fallen
        classification. Mask elongation (rotation-invariant) wins when available;
        box aspect is the fallback."""
        paired = list(zip(boxes, elongations or [None] * len(boxes)))
        paired = [(b, e) for b, e in paired
                  if (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) >= self.min_area_px]
        big = [b for b, _e in paired]
        if not big:
            return {"found": False, "area": 0.0, "point": None, "bbox": None, "count": 0,
                    "standing_count": 0, "fallen_count": 0, "pose": None}
        poses = []
        for b, e in paired:
            if e is None:
                w = max(1.0, float(b[2]) - float(b[0]))
                h = max(1.0, float(b[3]) - float(b[1]))
                e = max(w, h) / min(w, h)
            poses.append("fallen" if e >= self.FALLEN_ELONG else "standing")
        first_standing = next((b for b, p in zip(big, poses) if p == "standing"), None)
        primary = self._box_result(*(first_standing if first_standing is not None else big[0]))
        primary["count"] = len(big)
        primary["standing_count"] = poses.count("standing")
        primary["fallen_count"] = poses.count("fallen")
        primary["pose"] = ("standing" if first_standing is not None else poses[0]) \
            if primary.get("found") else None
        primary["boxes"] = [list(map(float, b)) for b in big]
        return primary

    @staticmethod
    def _mask_elongation(mask: np.ndarray) -> Optional[float]:
        """Major/minor axis ratio of a boolean mask via PCA of its pixel coordinates.
        Rotation-INVARIANT — a diagonally-lying tube has a square bounding box (the
        f635 blind spot) but an elongated mask."""
        ys, xs = np.nonzero(mask)
        if len(xs) < 20:
            return None
        pts = np.stack([xs, ys]).astype(np.float64)
        cov = np.cov(pts - pts.mean(axis=1, keepdims=True))
        w = np.sort(np.linalg.eigvalsh(cov))
        if w[-1] <= 0 or w[0] <= 1e-6:
            return None
        return float(np.sqrt(w[-1] / w[0]))

    # SAM3 two-prompt strategy, VALIDATED on real YAM wrist frames 2026-07-07 (all 4
    # verification cases correct — standing top-down, standing mid-table, fallen
    # diagonal, empty table):
    #  - CAP prompt -> STANDING candidates: segmenting the cap alone avoids the
    #    shadow, which the full-vial mask absorbed (elong 2.7 -> false "fallen").
    #  - TUBE prompt -> FALLEN candidates: the translucent lying tube is missed by
    #    "vial"-style prompts entirely; elong >= FALLEN_ELONG required (filters
    #    spurious matches).
    #  - dedup: a fallen vial's cap is also compact — a cap centered inside a tube
    #    instance is the same (fallen) vial, not a standing one.
    #  - frames < 400 px (the 224 pre-RPC obs) are 3x LANCZOS-upsampled first: the
    #    ~13 px top-down cap is invisible to SAM3 at 224.
    # "black round cap", NOT "black cap of a vial": with the gripper fingers flanking
    # the cap (the selector's pre-grasp confirmation view!) SAM3 scored the
    # vial-referencing prompt ~0.00 — the vial body is hidden, so the concept fails —
    # while the plain cap prompt scores 0.07-0.68 across all validated cases
    # (audit frame 1783480530.22: cap segmented perfectly, rejected on score).
    _SAM3_CAP_PROMPT = "black round cap"
    _SAM3_TUBE_PROMPT = "transparent glass tube lying on the table"

    def _sam3_seg(self, rgb: np.ndarray, prompt: str):
        """One /segment call -> [(box, score, mask_elong, area_frac)]."""
        import requests
        r = requests.post(f"{self.sam3_url.rstrip('/')}/segment", timeout=self.request_timeout_s,
                          json={"image_base64": _cv2_png_data_url(rgb).split(",", 1)[1],
                                "text_prompt": prompt})
        r.raise_for_status()
        out = []
        for res in r.json().get("results") or []:
            if not res.get("box"):
                continue
            e, area = None, 0.0
            try:
                if res.get("mask_base64") and res.get("shape"):
                    h, w = int(res["shape"][0]), int(res["shape"][1])
                    m = np.frombuffer(base64.b64decode(res["mask_base64"]),
                                      dtype=np.uint8).reshape(h, w) > 0
                    e = self._mask_elongation(m)
                    area = float(m.sum()) / float(h * w)
            except Exception:
                e, area = None, 0.0
            out.append((res["box"], float(res.get("score", 0.0)), e, area))
        return out

    def _sam3(self, rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """SAM 3 backend (CaP-X server schema). Two text prompts + geometric mask
        rules -> standing/fallen/no-vial. See the strategy comment above."""
        H, W = rgb.shape[:2]
        scale = 3 if max(H, W) < 400 else 1
        if scale > 1:
            import cv2
            img = cv2.resize(np.asarray(rgb, np.uint8), (W * scale, H * scale),
                             interpolation=cv2.INTER_CUBIC)
        else:
            img = rgb
        caps = [(b, s, e or 1.0) for b, s, e, a in self._sam3_seg(img, self._SAM3_CAP_PROMPT)
                if s >= 0.05 and a <= 0.03 and (e or 1.0) < self.FALLEN_ELONG]
        tubes = [(b, s, e or 99.0) for b, s, e, a in self._sam3_seg(img, self._SAM3_TUBE_PROMPT)
                 if s >= 0.25 and a <= 0.10 and (e or 0.0) >= self.FALLEN_ELONG]

        def _inside(cb, tb):
            cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
            return tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]
        caps = [c for c in caps if not any(_inside(c[0], t[0]) for t in tubes)]

        f = 1.0 / scale                       # boxes back in ORIGINAL frame coords
        boxes = [[v * f for v in b] for b, _s, _e in caps + tubes]
        elongs = [e for _b, _s, e in caps + tubes]
        return self._boxes_to_pose_result(boxes, elongs)

    def _parse_norm_boxes(self, txt: str, W: int, H: int) -> Dict[str, Any]:
        """Parse the shared contract ([{"box_2d":[ymin,xmin,ymax,xmax], "pose":...}]
        normalized 0-1000) into the detector result dict, scaled to the ORIGINAL image
        dims. JSON-first (carries the pose classification); regex fallback for models
        that return bare coordinate lists (then pose is unknown -> None)."""
        entries: list[tuple[list[float], Optional[str]]] = []
        try:
            import json as _json
            m = re.search(r'\[.*\]', txt, re.S)
            for e in (_json.loads(m.group(0)) if m else []):
                bb = e.get("box_2d") if isinstance(e, dict) else None
                if bb and len(bb) == 4:
                    pose = str(e.get("pose", "")).lower() or None
                    entries.append(([float(v) for v in bb], pose))
        except Exception:
            entries = []
        if not entries:
            entries = [([float(v) for v in row], None) for row in
                       re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', txt)]
        if not entries:
            return {"found": False, "area": 0.0, "point": None, "bbox": None, "count": 0,
                    "standing_count": 0, "fallen_count": 0, "pose": None}
        boxes, poses = [], []
        for (ymin, xmin, ymax, xmax), pose in entries:
            boxes.append([xmin / 1000 * W, ymin / 1000 * H, xmax / 1000 * W, ymax / 1000 * H])
            poses.append(pose)
        big = [(b, p) for b, p in zip(boxes, poses)
               if (b[2] - b[0]) * (b[3] - b[1]) >= self.min_area_px]
        # primary = first STANDING vial when poses are known (the graspable target)
        first_standing = next((b for b, p in big if p == "standing"), None)
        primary = self._box_result(*(first_standing if first_standing is not None else boxes[0]))
        primary["count"] = len(big)
        # pose counts are None (UNKNOWN) when the model returned bare coordinates —
        # 0 would read as "pose-judged, nothing standing" and wrongly block RL entry.
        have_pose = any(p is not None for _b, p in big)
        primary["standing_count"] = sum(1 for _b, p in big if p == "standing") if have_pose else None
        primary["fallen_count"] = sum(1 for _b, p in big if p == "fallen") if have_pose else None
        primary["pose"] = next((p for b, p in big
                                if primary["bbox"] and b == primary["bbox"]), None) \
            if primary.get("found") else None
        primary["boxes"] = boxes
        return primary

    def _downscale(self, rgb: np.ndarray, max_px: int = 320) -> np.ndarray:
        H, W = rgb.shape[:2]
        if max(H, W) > max_px:
            import cv2
            s = max_px / float(max(H, W))
            return cv2.resize(np.asarray(rgb, np.uint8),
                              (int(round(W * s)), int(round(H * s))),
                              interpolation=cv2.INTER_AREA)
        return rgb

    def _vlm_text(self, prompt: str, rgb: np.ndarray, backend: str | None = None) -> Optional[str]:
        """One prompt+image round-trip on a VLM backend (default: the main backend) ->
        response text (None when the backend/key is unavailable)."""
        import requests
        if (backend or self.backend) in ("gpt", "openai"):
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                p = pathlib.Path("/home/ssc/Desktop/research/.api/openai_key.txt")
                key = p.read_text().strip() if p.exists() else None
            if not key:
                return None
            # 512 px (vs Gemini's 320): from the head camera's wide shot a single
            # vial is ~15 px at 640 — 320 px halves that below detectability.
            payload = {"model": self.model or _GPT_MODEL,
                       "max_completion_tokens": self.max_tokens,
                       "messages": [{"role": "user", "content": [
                           {"type": "text", "text": prompt},
                           {"type": "image_url",
                            "image_url": {"url": _jpeg_data_url(self._downscale(rgb, 512))}}]}]}
            r = requests.post(_OPENAI_URL, json=payload, timeout=self.request_timeout_s,
                              headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        if not self._key:
            return None
        payload = {"model": self.model or _ER_MODEL, "temperature": 0.0, "max_tokens": self.max_tokens,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": prompt},
                       {"type": "image_url",
                        "image_url": {"url": _jpeg_data_url(self._downscale(rgb))}}]}]}
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.post(_LITELLM_URL, json=payload, verify=False, timeout=self.request_timeout_s,
                          headers={"Authorization": f"Bearer {self._key}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    def _release_vlm(self) -> Optional[str]:
        """Which VLM answers the SEMANTIC release query. Detector-only main backends
        (sam3 / sam3_owlvit) route to release_backend, auto-resolved from available
        keys; VLM main backends answer it themselves."""
        if self.backend not in ("sam3", "sam3_owlvit"):
            return self.backend
        if self.release_backend:
            return self.release_backend
        # user rule 2026-07-07: keep GEMINI-ER for the semantic lane by default
        if self._key:
            return "gemini_er"
        if os.environ.get("OPENAI_API_KEY") or \
                pathlib.Path("/home/ssc/Desktop/research/.api/openai_key.txt").exists():
            return "gpt"
        return None

    def _release(self, rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """Backend call for check_release: {'upright','near_table'} bools or None."""
        vlm = self._release_vlm()
        if vlm is None:
            return None                            # no VLM available for semantic queries
        txt = self._vlm_text(_RELEASE_PROMPT, rgb, backend=vlm)
        if not txt:
            return None
        m = re.search(r'\{.*\}', txt, re.S)
        if not m:
            return None
        try:
            import json as _json
            d = _json.loads(m.group(0))
        except Exception:
            return None
        return {"upright": bool(d.get("upright")), "near_table": bool(d.get("near_table"))}

    def _held(self, rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """Backend call for check_held: {'held': bool} or None."""
        vlm = self._release_vlm()              # same semantic lane as check_release
        if vlm is None:
            return None
        txt = self._vlm_text(_HELD_PROMPT, rgb, backend=vlm)
        if not txt:
            return None
        m = re.search(r'\{.*\}', txt, re.S)
        if not m:
            return None
        try:
            import json as _json
            d = _json.loads(m.group(0))
        except Exception:
            return None
        return {"held": bool(d.get("held"))}

    def _gpt(self, rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """GPT-5.5 backend (user option 2026-07-06): the OpenAI API directly, same
        prompt + box contract as Gemini-ER. Stronger scene understanding; per-call
        latency is higher, which only staleness-costs us — the background worker and
        per-camera cache keep the 30 Hz loop untouched."""
        H, W = rgb.shape[:2]                       # ORIGINAL dims (box coords scale to these)
        txt = self._vlm_text(_POINT_PROMPT, rgb)
        if txt is None:
            return None
        return self._parse_norm_boxes(txt, W, H)

    def _box_result(self, x0, y0, x1, y1) -> Dict[str, Any]:
        area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
        if area < self.min_area_px:
            return {"found": False, "area": float(area), "point": None, "bbox": None}
        return {"found": True, "area": float(area),
                "point": [float((x0 + x1) / 2), float((y0 + y1) / 2)],
                "bbox": [float(x0), float(y0), float(x1), float(y1)]}

    def _gemini(self, rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        H, W = rgb.shape[:2]                       # ORIGINAL dims (box coords scale to these)
        txt = self._vlm_text(_POINT_PROMPT, rgb)
        if txt is None:
            return None
        return self._parse_norm_boxes(txt, W, H)

    def _owlvit(self, rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        import requests
        r = requests.post(f"{self.owlvit_url.rstrip('/')}/detect", timeout=self.request_timeout_s,
                          json={"image_base64": _png_data_url(rgb).split(",", 1)[1],
                                "queries": ["a glass vial", "a small test tube"], "threshold": 0.1})
        r.raise_for_status()
        dets = r.json().get("detections") or r.json().get("boxes") or []
        if not dets:
            return {"found": False, "area": 0.0, "point": None, "bbox": None}
        b = dets[0].get("box") or dets[0].get("bbox") or dets[0]
        return self._box_result(b[0], b[1], b[2], b[3])


_SHARED: Dict[str, VialDetector] = {}


def make_vial_detector(backend: str | None = None, throttle_s: float = 1.0,
                       shared: bool = True, **kw) -> VialDetector:
    """Config factory. `shared=True` (default) returns ONE detector per backend across
    the whole agent — the verifier and the reset previously instantiated separate
    detectors (2 worker threads x 3 cameras of PNG+TLS in the agent process), which
    GIL-starved the 30 Hz control loop (observed 8-16 Hz on-robot 2026-07-06). Sharing
    also shares the per-camera cache, halving Gemini API calls."""
    key = backend or os.environ.get("SUBRL_PERCEPTION_BACKEND", "gemini_er")
    if shared:
        if key not in _SHARED:
            _SHARED[key] = VialDetector(backend=backend, throttle_s=throttle_s, **kw)
            try:
                from loguru import logger
                d = _SHARED[key]
                logger.info("[perception] detection backend: {} | semantic lane "
                            "(release/held checks): {}", d.backend, d._release_vlm())
            except Exception:
                pass
        return _SHARED[key]
    return VialDetector(backend=backend, throttle_s=throttle_s, **kw)


def observe_depth(obs) -> None:
    """Agent per-tick hook (depth mode 2026-08-05): fan the obs cameras' aligned
    depth frames into every shared detector that wants them. One call site
    (SubtaskRLAgent.act) instead of a depth kwarg threaded through every
    artifact detect_vial call."""
    for d in _SHARED.values():
        d.observe_depth(obs)
