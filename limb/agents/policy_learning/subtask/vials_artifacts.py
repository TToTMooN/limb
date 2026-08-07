"""Glue: load the coding-agent-authored vials-grasp artifacts (verifiers / selector
/ reset) and inject their runtime primitives, exposed as factories the limb config
can instantiate via `_target_`.

The authored code (in SubRL-VLA/real_robot/artifacts/vials_grasp/) calls global
perception primitives; we inject them from that dir's `primitives.py`. The proprio
primitives read straight from obs (gripper width + motor-current contact load + ee
lift) — the ONLY calibration is the table height, passed as `table_z`.

Verifiers + selector are fully wired here (pending only `table_z`). The authored
RESET additionally needs CONTROL primitives (move_right_ee_to / open/close gripper)
backed by the limb robot — pass them via `control=` once available; until then the
config uses the HoldReset placeholder.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
from typing import Any, Callable, Dict, Optional

DEFAULT_DIR = pathlib.Path(os.environ.get(
    "VIALS_ARTIFACTS_DIR",
    "/home/ssc/Desktop/research/SubRL-VLA/real_robot/artifacts/vials_grasp",
))


def _primitives(artifacts_dir: pathlib.Path, table_z: float,
                vial_visible_fn: Optional[Callable] = None,
                detector: Optional[Any] = None,
                confirm_detector: Optional[Any] = None,
                side: str = "right") -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("vials_primitives", artifacts_dir / "primitives.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    vv = vial_visible_fn or (detector.vial_visible if detector is not None else None)
    dv = detector.detect_vial if detector is not None else None
    rc = getattr(detector, "check_release", None) if detector is not None else None
    hc = getattr(detector, "check_held", None) if detector is not None else None
    cv = confirm_detector.detect_vial if confirm_detector is not None else None
    ic = getattr(detector, "check_inserted", None) if detector is not None else None
    sc = getattr(detector, "check_stand", None) if detector is not None else None
    return mod.primitives_dict(table_z=table_z, vial_visible_fn=vv, detect_vial_fn=dv,
                               release_check_fn=rc, held_check_fn=hc, confirm_detect_fn=cv,
                               insert_check_fn=ic, stand_check_fn=sc, side=side)


def _load_class(artifacts_dir: pathlib.Path, filename: str, class_name: str,
                inject: Dict[str, Any]) -> type:
    ns = dict(inject)
    code = (artifacts_dir / filename).read_text()
    exec(compile(code, filename, "exec"), ns)         # authored code-as-policy, by design
    return ns[class_name]


def make_verifiers(table_z: float = 0.0, stable_frames: int = 30, lift_m: float = 0.04,
                   artifacts_dir: Optional[str] = None, vial_visible_fn: Optional[Callable] = None,
                   detector: Optional[Any] = None, success_image_confirm: bool = True,
                   fallen_terminates: bool = True,
                   class_name: str = "VialsGraspVerifiers", side: str = "right", **ctor_kw):
    """SubtaskVerifiers from the authored verifiers.py. `table_z` = table height in the ee
    frame (the one on-robot calibration). `detector` = a VialDetector (image backend).
    `fallen_terminates=False` disables the verifier-confirmed-fallen episode termination
    (fallen then only blocks the HUMAN->RL resume; rollouts end via success/timeout).

    `class_name`/`side`/`**ctor_kw` (2026-07-31, insert sub-task): other artifact dirs
    author their own verifier class with its own ctor — e.g. vials_insert's
    VialsInsertVerifiers(side=..., stand_region=..., ...) — the grasp-shaped kwargs
    above only apply to the default grasp class."""
    d = pathlib.Path(artifacts_dir or DEFAULT_DIR)
    cls = _load_class(d, "verifiers.py", class_name,
                      _primitives(d, table_z, vial_visible_fn, detector, side=side))
    if class_name == "VialsGraspVerifiers":
        return cls(stable_frames=stable_frames, lift_m=lift_m,
                   success_image_confirm=success_image_confirm,
                   fallen_terminates=fallen_terminates)
    return cls(side=side, **ctor_kw)


def make_selector(table_z: float = 0.0, start_frames: int = 3, success_frames: int = 30,
                  lift_m: float = 0.05, center_frac: float = 0.35,
                  approach_z_min: Optional[float] = None, approach_z_max: Optional[float] = None,
                  region: Optional[tuple] = None, strict_wrist: bool = True,
                  artifacts_dir: Optional[str] = None,
                  vial_visible_fn: Optional[Callable] = None, detector: Optional[Any] = None,
                  confirm_detector: Optional[Any] = None, confirm_veto: bool = False,
                  stand_veto: bool = False,
                  class_name: str = "VialsGraspSelector", side: str = "right", **ctor_kw):
    """Image-grounded coding-agent SELECTOR: VLA approach -> 'rl' when the vial is
    centered in the RIGHT WRIST camera (gripper above it); RL -> 'vla' at inference
    after a stable hold. Requires the image `detector` (fail-safe: no detector -> no
    switch). Image confirmation is MANDATORY for every handoff (user rule 2026-07-07:
    never proprio-only); the selector warms the wrist-camera query from the first
    approach tick so slow backends have their answer cached in time.

    `confirm_detector` (user 2026-07-27, eval cascade): optional SECOND detector for
    the semantic check the primary can't make — SAM3 stays the fast primary lane
    (same as training) and the confirm backend (Gemini-ER, whose prompt already
    excludes racked/inserted/held vials) must also see a standing TABLE vial in the
    wrist view before VLA -> RL commits. No answer / outage -> the switch is blocked
    and the VLA keeps the task (existing fail-safe)."""
    d = pathlib.Path(artifacts_dir or DEFAULT_DIR)
    cls = _load_class(d, "selector.py", class_name,
                      _primitives(d, table_z, vial_visible_fn, detector, confirm_detector, side=side))
    if class_name == "VialsGraspSelector":
        return cls(start_frames=start_frames, success_frames=success_frames, lift_m=lift_m,
                   center_frac=center_frac,
                   approach_z_min=0.03 if approach_z_min is None else approach_z_min,
                   approach_z_max=0.18 if approach_z_max is None else approach_z_max,
                   region=region, strict_wrist=strict_wrist, confirm_veto=confirm_veto,
                   stand_veto=stand_veto)
    # replay-verify MINOR #5 (2026-07-31): approach_z_* are NAMED params of this
    # factory, so yaml values landed here and were silently DROPPED for non-grasp
    # classes. Forward them only when explicitly set (None-sentinel), so each
    # class's own defaults win when the yaml omits them.
    if approach_z_min is not None:
        ctor_kw["approach_z_min"] = approach_z_min
    if approach_z_max is not None:
        ctor_kw["approach_z_max"] = approach_z_max
    return cls(side=side, start_frames=start_frames, **ctor_kw)


def make_escalator(notify: Optional[Callable[[str], None]] = None):
    """RoboClaw 'Call Human' escalator. `notify` = optional side-channel (pedal-light,
    Slack/MCP notification, UI banner) called with the message on an unrecoverable state."""
    from .interfaces import LoggingEscalator
    return LoggingEscalator(notify=notify)


def make_reset(table_z: float = 0.0, control: Any = "ik_servo",
               artifacts_dir: Optional[str] = None, vial_visible_fn: Optional[Callable] = None,
               detector: Optional[Any] = None, seed: Optional[int] = None,
               step_rad: float = 0.02, gripper_open_cmd: float = 2.2):
    """ResetPolicy (EAP inverse) from the authored reset_policy.py. Places the HELD vial at a
    RANDOM reachable (x,y) on the table (per-episode randomized init).

    `control`:
      - "ik_servo" (default): NON-BLOCKING per-tick IK-servo primitives (review H16) — one
        damped-least-squares step per control tick, returned through the normal action path.
      - a dict of callables {move_right_ee_to, open_right_gripper, close_right_gripper, hold}
        for custom/robot-specific primitives.
    `seed=None` -> OS-entropy placement randomization."""
    d = pathlib.Path(artifacts_dir or DEFAULT_DIR)
    prims = _primitives(d, table_z, vial_visible_fn, detector)
    if control == "ik_servo":
        from .reset_control import ResetControl, TickWrappedReset
        ctl = ResetControl(step_rad=step_rad, gripper_open_cmd=gripper_open_cmd,
                           z_floor=table_z - 0.01)
        cls = _load_class(d, "reset_policy.py", "VialsGraspReset", {**prims, **ctl.primitives()})
        return TickWrappedReset(inner=cls(seed=seed), control=ctl)
    if not isinstance(control, dict) or not control:
        raise RuntimeError("make_reset: control must be 'ik_servo' or a dict of primitives")
    cls = _load_class(d, "reset_policy.py", "VialsGraspReset", {**prims, **control})
    return cls(seed=seed)
