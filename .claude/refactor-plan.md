# Refactor Plan: YAM-only Minimal Codebase

## Goal

Strip the repo to only what's needed for YAM bimanual arms:
- Real-hardware teleoperation (Viser, GELLO, VR)
- Camera observation collection (RealSense, ZED, OpenCV)
- Point cloud / depth processing
- VLA policy deployment (π0, diffusion)

## Files to Delete

### Franka robot driver & support
- [x] `limb/robots/franka_osc.py`
- [x] `limb/robots/robotiq_gripper.py`
- [x] `limb/robots/inverse_kinematics/franka_pyroki.py`
- [x] `limb/agents/teleoperation/franka_pyroki_viser_agent.py`
- [x] `limb/agents/teleoperation/franka_pyroki_viser_agent_linear_interp.py`
- [x] `limb/agents/client/` (entire folder)
- [x] `limb/serving/` (entire folder)
- [x] `configs/franka/` (entire folder)
- [x] `robot_configs/franka/` (entire folder)

### Sim-only code (MuJoCo / MJLab)
- [x] `limb/robots/mujoco_sim_robot.py`
- [x] `limb/robots/mjlab_sim_robot.py`
- [x] `limb/robots/yam_pick_red_cube_sim_robot.py`
- [x] `limb/mujoco/` (entire folder — assets, envs, convert_urdf)
- [x] `configs/yam/yam_gello_mujoco_sim.yaml`
- [x] `configs/yam/yam_gello_mjlab_sim.yaml`
- [x] `configs/yam/yam_gello_pick_red_cube_sim.yaml`
- [x] `configs/yam/yam_bilateral_mjlab_sim.yaml`

### Sim-only agents
- [x] `limb/agents/teleoperation/gello_leader_agent.py` (Feetech STS, sim only)
- [x] `limb/agents/teleoperation/bilateral_leader_agent.py` (haptic sim only)

### pyproject.toml cleanup
- [x] Removed `franka_panda` optional dependency
- [x] Removed `panda_python` uv source
- [x] Removed `minimalmodbus` (Robotiq-only)
- [x] Removed `dm_control` (sim-only, not needed for dm-env)

### Code fixes
- [x] Verified no broken imports remain

### Kept (still in use)
- `limb/input_devices/` (JoyCon gripper for GELLO teleop)
- `evdev` dependency (required by JoyCon)

## Status: Phase 1+2 COMPLETE

All deletions done on branch `refactor/yam-only`.
