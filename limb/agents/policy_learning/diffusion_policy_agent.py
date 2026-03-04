# type: ignore
import collections
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from dm_env.specs import Array

from limb.agents.agent import PolicyAgent
from limb.agents.constants import ActionSpec
from limb.utils.portal_utils import remote


def _recursive_flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict with ``-`` as the separator."""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}-{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_recursive_flatten(v, key))
        else:
            flat[key] = v
    return flat


def _reverse_flatten(d: Dict[str, Any], sep: str = "-") -> Dict[str, Any]:
    """Unflatten a dict whose keys were joined with *sep*."""
    result: Dict[str, Any] = {}
    for k, v in d.items():
        parts = k.split(sep)
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = v
    return result


class AsyncDiffusionAgent(PolicyAgent):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def load_model(self, folder_path: Union[str, Path], step: Optional[int] = None, bfloat16: bool = False) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def obs_to_model_input(self):
        raise NotImplementedError

    @remote()
    def act(self, obs):
        action = _reverse_flatten(self(obs))["action"]

        return {
            "left": {"pos": action["left"]["pos"]},
            "right": {"pos": action["right"]["pos"]},
        }

    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        """Define the action specification."""
        if self.use_joint_state_as_action:
            return {
                "left": {
                    "pos": Array(shape=(7,), dtype=np.float32),
                    "vel": Array(shape=(7,), dtype=np.float32),
                },
                "right": {
                    "pos": Array(shape=(7,), dtype=np.float32),
                    "vel": Array(shape=(7,), dtype=np.float32),
                },
            }
        else:
            return {
                "left": {"pos": Array(shape=(7,), dtype=np.float32)},
                "right": {"pos": Array(shape=(7,), dtype=np.float32)},
            }

    def __call__(self, obs):
        obs = _recursive_flatten(obs)
        with self.obs_deque_lock:
            if self.obs_deque is None:
                self.obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)
            # add latest observation to deque
            self.obs_deque.append(obs)
        raise NotImplementedError
