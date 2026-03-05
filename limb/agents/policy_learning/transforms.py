"""Observation and action transforms for policy agents.

YAML-configurable dataclasses that map between limb's Observation format
and what policy servers expect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _recursive_flatten(d: Dict[str, Any], prefix: str = "", sep: str = "-") -> Dict[str, Any]:
    """Flatten a nested dict with the given separator."""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_recursive_flatten(v, key, sep))
        else:
            flat[key] = v
    return flat


@dataclass
class ObsTransform:
    """Transform limb observations into the format expected by a generic policy server.

    Handles flattening nested obs dict, concatenating state keys, resizing images,
    and adding optional language prompts.

    Parameters
    ----------
    state_keys : list of str
        Flattened observation keys to concatenate into the "state" vector.
        Example: ["left-joint_pos", "left-gripper_pos", "right-joint_pos", "right-gripper_pos"]
    image_keys : dict mapping server_name -> obs_key
        Maps the server's expected image name to the flattened obs key.
        Example: {"left_camera": "left_wrist_camera-images-rgb"}
    image_size : (height, width)
        Resize all images to this size before sending.
    image_format : str
        "uint8_hwc" — (H, W, 3) uint8 (default, for generic server spec).
        "uint8_chw" — (3, H, W) uint8 (for OpenPI-style servers).
    prompt : str or None
        Language instruction to include in every observation.
    """

    state_keys: List[str] = field(default_factory=list)
    image_keys: Dict[str, str] = field(default_factory=dict)
    image_size: Tuple[int, int] = (224, 224)
    image_format: str = "uint8_hwc"
    prompt: Optional[str] = None

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        flat = _recursive_flatten(obs)

        state_parts = []
        for key in self.state_keys:
            val = flat[key]
            state_parts.append(np.atleast_1d(val).astype(np.float32))
        state = np.concatenate(state_parts) if state_parts else np.array([], dtype=np.float32)

        images = {}
        h, w = self.image_size
        for server_name, obs_key in self.image_keys.items():
            img = flat[obs_key]
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            if self.image_format == "uint8_chw":
                img = np.transpose(img, (2, 0, 1))
            images[server_name] = img

        result: Dict[str, Any] = {"state": state, "images": images}
        if self.prompt is not None:
            result["prompt"] = self.prompt
        return result


@dataclass
class OpenPIObsTransform:
    """Transform limb observations into OpenPI's expected flat-dict format.

    OpenPI expects a flat dict with keys like "left_camera-images-rgb"
    for images and "state" for the concatenated proprioceptive state.
    Images must be (3, H, W) uint8 with padded resize.
    """

    state_keys: List[str] = field(
        default_factory=lambda: [
            "left-joint_pos",
            "left-gripper_pos",
            "right-joint_pos",
            "right-gripper_pos",
        ]
    )
    image_keys: List[str] = field(
        default_factory=lambda: [
            "left_wrist_camera-images-rgb",
            "right_wrist_camera-images-rgb",
            "head_camera-images-rgb",
        ]
    )
    image_size: Tuple[int, int] = (224, 224)

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        flat = _recursive_flatten(obs)

        state_parts = [flat[k] for k in self.state_keys]
        state = np.concatenate(state_parts, axis=-1).astype(np.float32)

        from openpi_client import image_tools

        h, w = self.image_size
        result: Dict[str, Any] = {"state": state}
        for key in self.image_keys:
            if key not in flat:
                continue
            img = flat[key]
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, h, w))
            img = np.transpose(img, (2, 0, 1))
            result[key] = img

        return result


@dataclass
class ActionTransform:
    """Transform raw action arrays from a policy server into limb's action dict.

    Splits a flat action vector into per-arm segments with gripper clipping.

    Parameters
    ----------
    arm_names : list of str
        Arm names in order, e.g. ["left", "right"].
    joints_per_arm : int
        Number of joints per arm (including gripper). Default 7 (6 arm + 1 gripper).
    include_vel : bool
        If True, the action vector has pos AND vel for each arm.
    gripper_clip : (min, max)
        Clip gripper to this range. Default (0.0, 1.0).
    """

    arm_names: List[str] = field(default_factory=lambda: ["left", "right"])
    joints_per_arm: int = 7
    include_vel: bool = False
    gripper_clip: Tuple[float, float] = (0.0, 1.0)

    def __call__(self, action_array: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        stride = self.joints_per_arm * (2 if self.include_vel else 1)
        result = {}
        for i, name in enumerate(self.arm_names):
            segment = action_array[i * stride : (i + 1) * stride]
            pos = segment[: self.joints_per_arm].copy()
            pos[-1] = np.clip(pos[-1], *self.gripper_clip)
            arm_action: Dict[str, np.ndarray] = {"pos": pos}
            if self.include_vel:
                vel = segment[self.joints_per_arm :]
                arm_action["vel"] = vel
            result[name] = arm_action
        return result
