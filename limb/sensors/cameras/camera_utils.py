"""Camera utility functions for processing observation data.

Generic image ops (resize_with_pad, resize_with_center_crop) are in ``robocam.utils``.
This module provides limb-specific observation extraction helpers.
"""

from typing import Any, Dict

import numpy as np

# Re-export generic image utils from robocam
from robocam.utils import resize_with_center_crop, resize_with_pad  # noqa: F401


def obs_get_rgb(obs: Any) -> Dict[str, np.ndarray]:
    """Extract RGB images from an observation (typed ``Observation`` or legacy dict)."""
    from limb.core.observation import Observation

    if isinstance(obs, Observation):
        return {name: cam.rgb for name, cam in obs.cameras.items()}

    rgb_dict: Dict[str, np.ndarray] = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            if "images" in value and isinstance(value["images"], dict):
                if "rgb" in value["images"]:
                    rgb_dict[key] = value["images"]["rgb"]
                elif "left_rgb" in value["images"]:
                    rgb_dict[key] = value["images"]["left_rgb"]
                elif "right_rgb" in value["images"]:
                    rgb_dict[key] = value["images"]["right_rgb"]
            else:
                nested_rgb = obs_get_rgb(value)
                rgb_dict.update(nested_rgb)
    return rgb_dict


def obs_get_camera_data(obs: Any) -> Dict[str, Dict[str, Any]]:
    """Extract all camera data from an observation (typed ``Observation`` or legacy dict)."""
    from limb.core.observation import Observation

    if isinstance(obs, Observation):
        result: Dict[str, Dict[str, Any]] = {}
        for name, cam in obs.cameras.items():
            images: Dict[str, np.ndarray] = {"rgb": cam.rgb}
            if cam.depth is not None:
                images["depth"] = cam.depth
            if cam.left_rgb is not None:
                images["left_rgb"] = cam.left_rgb
            if cam.right_rgb is not None:
                images["right_rgb"] = cam.right_rgb
            entry: Dict[str, Any] = {"images": images, "timestamp": cam.timestamp}
            if cam.intrinsics is not None:
                entry["intrinsics"] = cam.intrinsics
            result[name] = entry
        return result

    camera_dict: Dict[str, Dict[str, Any]] = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            if "images" in value and "timestamp" in value:
                camera_dict[key] = value
            else:
                nested_cameras = obs_get_camera_data(value)
                camera_dict.update(nested_cameras)
    return camera_dict


def obs_has_cameras(obs: Any) -> bool:
    """Check if observation contains any camera data."""
    from limb.core.observation import Observation

    if isinstance(obs, Observation):
        return len(obs.cameras) > 0
    return len(obs_get_camera_data(obs)) > 0
