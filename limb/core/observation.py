"""Typed observation dataclasses for the limb control stack.

These wrap the untyped dicts that flow over Portal RPC into structured types
used within the main process (robot_env, launch, viser_monitor).  Agents still
receive plain dicts via ``Observation.to_dict()`` at the portal boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ArmObservation:
    """Observation from a single robot arm (e.g. YAM motor chain)."""

    joint_pos: np.ndarray  # (n_joints,) — 6 for YAM
    joint_vel: np.ndarray  # (n_joints,)
    gripper_pos: Optional[np.ndarray] = None  # (1,)
    joint_eff: Optional[np.ndarray] = None  # torques
    ee_pose: Optional[np.ndarray] = None  # (7,) wxyz+xyz — filled by IK agents
    extra: Optional[Dict[str, Any]] = None  # temp_mos, temp_rotor, future sensors


@dataclass
class CameraObservation:
    """Observation from a single camera (RealSense, OpenCV, ZED)."""

    rgb: np.ndarray  # (H,W,3) — primary image (always present)
    timestamp: float
    depth: Optional[np.ndarray] = None  # (H,W) float32
    intrinsics: Optional[np.ndarray] = None  # (3,3)
    left_rgb: Optional[np.ndarray] = None  # ZED stereo left
    right_rgb: Optional[np.ndarray] = None  # ZED stereo right
    point_cloud: Optional[np.ndarray] = None  # (N,3)
    extra: Optional[Dict[str, Any]] = None


@dataclass
class Observation:
    """Top-level observation combining all arms and cameras."""

    timestamp: float
    arms: Dict[str, ArmObservation] = field(default_factory=dict)
    cameras: Dict[str, CameraObservation] = field(default_factory=dict)
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to nested dict for portal RPC / policy model input.

        Produces the same layout as the original untyped observation dict:
        ``{arm_name: {joint_pos: ..., ...}, camera_name: {images: ..., timestamp: ...}, timestamp: ...}``
        """
        d: Dict[str, Any] = {"timestamp": self.timestamp}

        for name, arm in self.arms.items():
            arm_d: Dict[str, Any] = {
                "joint_pos": arm.joint_pos,
                "joint_vel": arm.joint_vel,
            }
            if arm.gripper_pos is not None:
                arm_d["gripper_pos"] = arm.gripper_pos
            if arm.joint_eff is not None:
                arm_d["joint_eff"] = arm.joint_eff
            if arm.ee_pose is not None:
                arm_d["ee_pose"] = arm.ee_pose
            if arm.extra:
                arm_d.update(arm.extra)
            d[name] = arm_d

        for name, cam in self.cameras.items():
            images: Dict[str, np.ndarray] = {"rgb": cam.rgb}
            if cam.left_rgb is not None:
                images["left_rgb"] = cam.left_rgb
            if cam.right_rgb is not None:
                images["right_rgb"] = cam.right_rgb
            if cam.depth is not None:
                images["depth"] = cam.depth
            cam_d: Dict[str, Any] = {
                "images": images,
                "timestamp": cam.timestamp,
            }
            if cam.intrinsics is not None:
                cam_d["intrinsics"] = cam.intrinsics
            if cam.point_cloud is not None:
                cam_d["point_cloud"] = cam.point_cloud
            if cam.extra:
                cam_d.update(cam.extra)
            d[name] = cam_d

        if self.extra:
            d.update(self.extra)

        return d


# ------------------------------------------------------------------ #
#  Conversion helpers (dict → dataclass)
# ------------------------------------------------------------------ #


def arm_obs_from_dict(d: Dict[str, Any]) -> ArmObservation:
    """Convert an i2rt ``get_observations()`` result dict to ``ArmObservation``."""
    extra_keys = set(d.keys()) - {"joint_pos", "joint_vel", "gripper_pos", "joint_eff", "ee_pose"}
    extra = {k: d[k] for k in extra_keys} if extra_keys else None
    return ArmObservation(
        joint_pos=d["joint_pos"],
        joint_vel=d["joint_vel"],
        gripper_pos=d.get("gripper_pos"),
        joint_eff=d.get("joint_eff"),
        ee_pose=d.get("ee_pose"),
        extra=extra,
    )


def camera_obs_from_dict(d: Dict[str, Any]) -> CameraObservation:
    """Convert a ``CameraNode.read()`` result dict to ``CameraObservation``.

    Handles RealSense (rgb + depth), OpenCV (rgb only), and ZED (left_rgb / right_rgb) layouts.
    The dict has shape ``{images: {rgb: ..., depth: ...}, timestamp: ..., intrinsics: ...}``.
    """
    images = d.get("images", {})

    # Primary RGB: prefer "rgb", fall back to "left_rgb"
    rgb = images.get("rgb")
    if rgb is None:
        rgb = images.get("left_rgb")
    if rgb is None:
        raise ValueError(f"Camera dict has no 'rgb' or 'left_rgb' in images. Keys: {list(images.keys())}")

    # Intrinsics can come from the images sub-dict or the top-level dict
    intrinsics = d.get("intrinsics")
    if isinstance(intrinsics, dict):
        # Some cameras return a nested dict with a matrix inside
        intrinsics = intrinsics.get("intrinsic_matrix", intrinsics.get("K"))

    # Collect leftover keys into extra
    known_top = {"images", "timestamp", "intrinsics", "point_cloud"}
    extra_keys = set(d.keys()) - known_top
    extra = {k: d[k] for k in extra_keys} if extra_keys else None

    return CameraObservation(
        rgb=rgb,
        timestamp=d.get("timestamp", 0.0),
        depth=images.get("depth"),
        intrinsics=intrinsics,
        left_rgb=images.get("left_rgb"),
        right_rgb=images.get("right_rgb"),
        point_cloud=d.get("point_cloud"),
        extra=extra,
    )
