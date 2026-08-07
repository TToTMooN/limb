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


def _resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Aspect-preserving resize + black-pad to (target_h, target_w). Pure numpy + cv2.

    Mirrors openpi_client.image_tools.resize_with_pad so we don't pull in openpi_client
    just for one helper.
    """
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_h - new_h) // 2
    pad_bot = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return cv2.copyMakeBorder(resized, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)


@dataclass
class OpenPIObsTransform:
    """Transform limb observations into OpenPI's expected obs format.

    OpenPI's input transforms (AlohaInputs, LiberoInputs, DroidInputs, …) all read
    images from ``data["images"][<server_name>]``, NOT from top-level keys. Image
    tensors must be (3, H, W) uint8 with aspect-preserving padded resize.

    Image server-side names depend on the model's input policy:
      • AlohaInputs (pi0/pi0.5 aloha-style):  cam_high / cam_left_wrist / cam_right_wrist
      • LiberoInputs:                          image / wrist_image
      • DroidInputs:                           exterior_image_1_left / wrist_image_left

    Parameters
    ----------
    state_keys : list of str
        Flattened observation keys to concatenate into the "state" vector.
    image_keys : dict mapping server_name -> obs_key
        Maps the server's expected image name to the flattened limb obs key.
        Example: {"cam_high": "head_camera-images-rgb",
                  "cam_left_wrist": "left_wrist_camera-images-rgb",
                  "cam_right_wrist": "right_wrist_camera-images-rgb"}
    image_size : (height, width)
        Target image resolution for the model.
    prompt : str or None
        Language instruction sent under the "prompt" key on every observation.
        Required for VLA models (pi0, pi0.5, X-VLA, SmolVLA).
    adv_ind : str or None
        Advantage-conditioning token sent under the "adv_ind" key on every
        observation. Required for pistar / pi0.6 RECAP models (otherwise the
        server's TokenizePrompt raises "Adv_ind is required."). Standard
        serving-time value is "positive". Leave None for vanilla pi0 / pi0.5.
    """

    state_keys: List[str] = field(
        default_factory=lambda: [
            "left-joint_pos",
            "left-gripper_pos",
            "right-joint_pos",
            "right-gripper_pos",
        ]
    )
    image_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "cam_high": "head_camera-images-rgb",
            "cam_left_wrist": "left_wrist_camera-images-rgb",
            "cam_right_wrist": "right_wrist_camera-images-rgb",
        }
    )
    image_size: Tuple[int, int] = (224, 224)
    prompt: Optional[str] = None
    adv_ind: Optional[str] = None

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        flat = _recursive_flatten(obs)

        state_parts = [flat[k] for k in self.state_keys]
        state = np.concatenate(state_parts, axis=-1).astype(np.float32)

        h, w = self.image_size
        images: Dict[str, np.ndarray] = {}
        for server_name, obs_key in self.image_keys.items():
            if obs_key not in flat:
                continue
            img = flat[obs_key]
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img = _resize_with_pad(img, h, w)
            img = np.transpose(img, (2, 0, 1))
            images[server_name] = img

        # OpenPI input transforms (AlohaInputs etc.) read images from data["images"].
        result: Dict[str, Any] = {"state": state, "images": images}
        if self.prompt is not None:
            result["prompt"] = self.prompt
        if self.adv_ind is not None:
            result["adv_ind"] = self.adv_ind
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
