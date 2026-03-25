import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import viser
import viser.extras
from dm_env.specs import Array

from limb.agents.agent import Agent
from limb.utils.portal_utils import remote
from limb.visualization.panels.camera_panel import CameraPanel


def _create_ik_solver(solver_name: str, ik_params: Optional[Dict[str, Any]] = None, **kwargs):
    extra = ik_params or {}
    if solver_name == "pyroki":
        from limb.robots.inverse_kinematics.yam_pyroki import YamPyroki

        return YamPyroki(**kwargs)
    elif solver_name == "pink":
        from limb.robots.inverse_kinematics.yam_pink import YamPink

        return YamPink(**{**kwargs, **extra})
    else:
        raise ValueError(f"Unknown IK solver: {solver_name!r}. Choose 'pyroki' or 'pink'.")


class YamViserAgent(Agent):
    def __init__(
        self,
        bimanual: bool = False,
        right_arm_extrinsic: Optional[Dict[str, Any]] = None,
        ik_solver: str = "pink",
        ik_params: Optional[Dict[str, Any]] = None,
    ):
        self.right_arm_extrinsic = right_arm_extrinsic
        self.bimanual = bimanual
        if bimanual:
            assert right_arm_extrinsic is not None, "right_arm_extrinsic must be provided for bimanual robot"
        self.viser_server = viser.ViserServer()
        self.ik = _create_ik_solver(ik_solver, ik_params=ik_params, viser_server=self.viser_server, bimanual=bimanual)

        # Camera panel handles feed thumbnails on the same server
        self._camera_panel = CameraPanel()
        self._camera_panel.attach(self.viser_server)

        self.ik_thread = threading.Thread(target=self.ik.run)
        self.ik_thread.start()
        self.obs = None
        self.real_vis_thread = threading.Thread(target=self._update_visualization, daemon=True)
        self.real_vis_thread.start()
        self._setup_visualization()

    def _setup_visualization(self):
        self.base_frame_left_real = self.viser_server.scene.add_frame("/base_left_real", show_axes=False)
        self.urdf_vis_left_real = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/base_left_real",
            mesh_color_override=(0.8, 0.5, 0.5),
        )
        for mesh in self.urdf_vis_left_real._meshes:
            mesh.opacity = 0.25  # type: ignore
        self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
            "Left Gripper", min=0.0, max=2.4, step=0.01, initial_value=0.0
        )

        if self.bimanual and self.right_arm_extrinsic is not None:
            self.ik.base_frame_right.position = np.array(self.right_arm_extrinsic["position"])
            self.ik.base_frame_right.wxyz = np.array(self.right_arm_extrinsic["rotation"])
            self.base_frame_right_real = self.viser_server.scene.add_frame(
                "/base_left_real/base_right_real", show_axes=False
            )
            self.base_frame_right_real.position = self.ik.base_frame_right.position
            self.urdf_vis_right_real = viser.extras.ViserUrdf(
                self.viser_server,
                deepcopy(self.ik.urdf),
                root_node_name="/base_left_real/base_right_real",
                mesh_color_override=(0.8, 0.5, 0.5),
            )
            for mesh in self.urdf_vis_right_real._meshes:
                mesh.opacity = 0.25  # type: ignore
            self.right_gripper_slider_handle = self.viser_server.gui.add_slider(
                "Right Gripper", min=0.0, max=2.4, step=0.01, initial_value=0.0
            )

    def _update_visualization(self):
        """Update real robot state URDF overlay from observations."""
        while self.obs is None:
            time.sleep(0.025)
        while True:
            if self.bimanual:
                self.urdf_vis_right_real.update_cfg(np.flip(self.obs["right"]["joint_pos"]))
            self.urdf_vis_left_real.update_cfg(np.flip(self.obs["left"]["joint_pos"]))
            time.sleep(0.02)

    def act(self, obs: Dict[str, Any]) -> Any:
        self.obs = deepcopy(obs)

        # Feed camera images to camera panel
        self._camera_panel.update(obs)

        action = {
            "left": {
                "pos": np.concatenate([np.flip(self.ik.joints["left"]), [self.left_gripper_slider_handle.value]]),
            }
        }
        if self.bimanual:
            assert self.ik.joints.keys() == {"left", "right"}, (
                "bimanual mode must have both left and right joint ik solved"
            )
            action["right"] = {
                "pos": np.concatenate([np.flip(self.ik.joints["right"]), [self.right_gripper_slider_handle.value]]),
            }

        return action

    def close(self) -> None:
        self._camera_panel.detach()

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        """Define the action specification."""
        action_spec = {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
        }
        if self.bimanual:
            action_spec["right"] = {"pos": Array(shape=(7,), dtype=np.float32)}
        return action_spec
