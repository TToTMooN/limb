# YAM Hardware Notes

## Debug joint values

```bash
uv run python scripts/diagnostics/print_YAM_joints.py --channel can_name
```

Reads `DMChainCanInterface` directly, so it has no joint-limit guard and won't freeze even when the arm is currently in a fault state. Pass `--channel can_leader_l` / `can_leader_r` / `can_follow_l` / `can_follow_r` to pick the arm.

---

## Leader Follower Teleop
Gripper Spring Feedback 

τ = kp · (spring_target_rad − q)  +  kd · (0 − q̇)  + spring_torque_bias

# Arm Safety
## Joint limits

Three different limits exist on every joint. Knowing which is which makes the runtime errors easy to interpret.

| Layer                    | Source                                                            | Purpose                                   |
|--------------------------|-------------------------------------------------------------------|-------------------------------------------|
| **Mechanical hard stop** | Physical arm                                                      | Metal-on-metal travel boundary            |
| **YAML soft limit**      | `joint_limits:` in `robot_configs/yam/<arm>.yaml`                 | Per-deployment operating policy           |
| **i2rt-effective limit** | YAML ± `buffer_rad` (default `0.1` rad)                           | What `MotorChainRobot` actually enforces  |

The i2rt buffer is applied at runtime by `_check_current_qpos_in_joint_limits` (`dependencies/i2rt/i2rt/robots/motor_chain_robot.py:206`) — it widens the YAML limits on both sides:

```python
lower_limits = arm_limits[:, 0] - buffer_rad
upper_limits = arm_limits[:, 1] + buffer_rad
```

### Per-joint values

Followers run position-controlled (tight limits); leaders run zero-torque + grav-comp only, so leader limits are widened to roughly the mechanical range minus the i2rt buffer. Fill in the **Mechanical limit** column per physical arm using `print_YAM_joints.py` (back-drive each joint through its full range; the value is the encoder reading at the hard stop).

| Joint | Default YAM (`left.yaml`) | Leader Teleop Arm (`leader_left.yaml`) | Approx Mechanical limit |
|------:|:--------------------------|:---------------------------------------------------------|:-----------------|
| J1    | [-2.09, 3.14]             | [-2.40, 3.14]                                            | [-2.53, 3.22]    |
| J2    | [ 0.00, 3.14]             | [-0.10, 3.44]                                            | [ 0.00, 3.67]    |
| J3    | [ 0.05, 3.14]             | [-0.05, 3.14]                                            | [ 0.00, 3.14]    |
| J4    | [-1.35, 1.35]             | [-1.60, 1.60]                                            | [-1.62, 1.64]    |
| J5    | [-1.50, 1.50]             | [-1.50, 1.50]                                            | [-1.59, 1.54]    |
| J6    | [-2.00, 2.00]             | [-2.00, 2.00]                                            | [-2.00, 2.18]    |

### Bilateral teleop clamp

`YamYamBilateralAgent` clips leader joint positions to the follower's YAML limits before commanding the follower (`limb/agents/teleoperation/yam_yam_bilateral_agent.py:118`). Without this, a leader with widened limits would push the follower's measured qpos past its bound and trip i2rt's `RuntimeError`.

The clamp values are hardcoded in `_YAM_JOINT_LIMITS` (`:40-50`) as a self-contained mirror of `robot_configs/yam/left.yaml`. **Keep the two in sync when editing follower limits.** Other agents (GELLO, VR, Viser IK, policies) do not apply this clamp.

