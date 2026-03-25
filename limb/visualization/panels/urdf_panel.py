"""URDF panel — 3D robot visualization in the viser scene."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import viser
import viser.extras

from limb.core.observation import Observation


@dataclass
class URDFPanel:
    """Shows the YAM URDF in the viser 3D viewport, updated from observations.

    Implements ObservablePanel: attach(server), update(obs), detach().
    """

    bimanual: bool = False
    right_arm_extrinsic: Optional[Dict[str, Any]] = None

    _server: Optional[viser.ViserServer] = field(default=None, init=False, repr=False)
    _urdf_vis_left: Optional[viser.extras.ViserUrdf] = field(default=None, init=False, repr=False)
    _urdf_vis_right: Optional[viser.extras.ViserUrdf] = field(default=None, init=False, repr=False)

    def attach(self, server: viser.ViserServer) -> None:
        self._server = server
        self._setup_urdf()

    def _setup_urdf(self) -> None:
        import yourdfpy

        from limb import ROOT_PATH

        yam_models = os.path.join(ROOT_PATH, "dependencies", "i2rt", "i2rt", "robot_models", "yam")
        urdf_path = os.path.join(yam_models, "yam.urdf")
        mesh_dir = os.path.join(yam_models, "assets")

        urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)

        self._server.scene.add_frame("/base", show_axes=False)
        self._urdf_vis_left = viser.extras.ViserUrdf(self._server, urdf, root_node_name="/base")
        self._server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

        if self.bimanual and self.right_arm_extrinsic is not None:
            right_frame = self._server.scene.add_frame("/base/base_right", show_axes=False)
            right_frame.position = tuple(self.right_arm_extrinsic.get("position", [0, -0.61, 0]))
            if "rotation" in self.right_arm_extrinsic:
                right_frame.wxyz = tuple(self.right_arm_extrinsic["rotation"])
            self._urdf_vis_right = viser.extras.ViserUrdf(
                self._server, deepcopy(urdf), root_node_name="/base/base_right"
            )

    def update(self, obs: Any) -> None:
        """Update URDF joint visualization from observation."""
        if self._urdf_vis_left is None:
            return

        if isinstance(obs, Observation):
            left = obs.arms.get("left")
            left_jp = left.joint_pos if left is not None else None
            right_data = obs.arms.get("right")
            right_jp = right_data.joint_pos if right_data is not None else None
        else:
            left = obs.get("left")
            left_jp = left["joint_pos"] if isinstance(left, dict) and "joint_pos" in left else None
            right = obs.get("right")
            right_jp = right["joint_pos"] if isinstance(right, dict) and "joint_pos" in right else None

        if left_jp is not None:
            self._urdf_vis_left.update_cfg(np.flip(left_jp[:6]))
        if self.bimanual and self._urdf_vis_right is not None and right_jp is not None:
            self._urdf_vis_right.update_cfg(np.flip(right_jp[:6]))

    def detach(self) -> None:
        self._urdf_vis_left = None
        self._urdf_vis_right = None
