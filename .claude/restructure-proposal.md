# Restructure Proposal: Clean Architecture for YAM + VLA + Data Collection

## Current Problems

### 1. IK/Visualization Tangling (CRITICAL)

`ViserAbstractBase` mixes three concerns in one class:
- **Pure IK solving** (algorithm, headless)
- **Viser 3D visualization** (URDF rendering, camera feeds)
- **Gizmo interactivity** (drag-to-set-target input)

Both `YamPink` and `YamPyroki` inherit this base, meaning even a headless IK
call carries conditional `if self.viser_server is not None:` checks everywhere.

`YamViserAgent` then creates its own ViserServer + ViserMonitor + IK solver,
spawning two threads. This makes it hard to test, hard to reuse IK in policy
code, and hard to add new IK consumers.

### 2. No Observation Schema

Observations are untyped nested dicts. Agents hardcode expected keys like
`obs["left"]["joint_pos"]`. Policy agents flatten to different key schemes
(`"left-joint_pos"`, `"left_camera-images-rgb"`). No validation, easy to break.

### 3. No Data Collection Layer

- ViserMonitor records video but no structured trajectories
- No concept of episodes, demos, or replay buffers
- No standardized serialization for obs+action pairs

### 4. diffusion_policy_agent.py is Broken

Imports non-existent modules (`limb.data.data_utils`,
`limb.learning.diffusion_policy.policy_network`). Either fix or
delete and rebuild when actually needed.

### 5. Agent Protocol Too Thin

No `observation_spec()`, no `reset()`, no lifecycle management.
Teleop and policy agents have different needs but share one minimal protocol.

---

## Proposed Structure

```
limb/
  core/                        # NEW: shared primitives
    observation.py             # ObservationDict type, schema validation
    action.py                  # ActionDict type, spec
    episode.py                 # Episode container (obs+action sequences)

  ik/                          # EXTRACTED from robots/viser/ + inverse_kinematics/
    solver.py                  # IKSolver protocol (pure algorithm interface)
    pink_solver.py             # YamPink IK (no viser imports)
    pyroki_solver.py           # YamPyroki IK (no viser imports)
    pyroki_snippets/           # JAX utilities (keep)

  visualization/               # EXTRACTED from robots/viser/
    viser_server.py            # Shared ViserServer lifecycle management
    ik_visualizer.py           # IK target gizmos + URDF overlay (optional)
    monitor.py                 # Camera feeds + recording (was viser_monitor.py)

  agents/
    agent.py                   # Agent protocol (with obs_spec, action_spec, reset)
    teleoperation/
      viser_agent.py           # Composes ik/ + visualization/ (no thread spawning)
      gello_agent.py           # Direct joint mapping (unchanged, clean)
      vr_agent.py              # Composes ik/ headless + xr_client
    policy/                    # Renamed from policy_learning/
      pi0_agent.py             # π0 HTTP client (keep, works well)
      diffusion_agent.py       # Rebuild when needed (or fix imports)

  robots/
    robot.py                   # Robot protocol (keep)
    yam_robot.py               # Renamed from yam_motor_chain_robot.py
    utils.py                   # Rate, Timeout (keep)

  sensors/
    cameras/                   # Keep as-is (clean)
      camera.py
      realsense.py             # Renamed (drop _camera suffix)
      opencv.py
      zed.py
      utils.py                 # Renamed from camera_utils.py

  data/                        # NEW: data collection & replay
    recorder.py                # Episode recorder (obs+action to disk)
    replay_buffer.py           # Load recorded episodes
    formats.py                 # HDF5/zarr/LeRobot serialization

  devices/                     # Renamed from dynamixel/
    dynamixel_reader.py        # USB GELLO (keep)
    network_dynamixel_reader.py # TCP GELLO (keep)

  envs/
    launch.py                  # Main entry (simplified)
    robot_env.py               # dm_env wrapper (keep)
    configs/
      loader.py                # YAML loading (keep)
      instantiate.py           # _target_ → class (keep)

  utils/
    launch_utils.py            # CAN setup, init helpers
    portal_utils.py            # RPC (keep)
    xr_client.py               # VR input (keep)
    depth_utils.py             # Point cloud (keep)
```

### Key changes:

1. **`ik/`** — Pure IK solvers extracted from viser. No visualization imports.
   `IKSolver.solve(targets) -> joint_positions`. Testable standalone.

2. **`visualization/`** — Optional layer that wraps IK for interactive use.
   `IKVisualizer(server, solver)` adds gizmos. `Monitor(server)` adds camera feeds.

3. **`core/`** — Shared types. `ObservationDict` with schema validation.
   Agents declare what they need via `observation_spec()`.

4. **`data/`** — Episode recording + replay. Plugs into `RobotEnv.step()` as
   an observer. Records `(timestamp, obs, action)` tuples per episode.

5. **Agent protocol expanded:**
   ```python
   class Agent(Protocol):
       def observation_spec(self) -> ObservationSpec: ...
       def action_spec(self) -> ActionSpec: ...
       def act(self, obs: ObservationDict) -> ActionDict: ...
       def reset(self) -> None: ...
       def close(self) -> None: ...
   ```

---

## Migration Path (Incremental)

### Phase A: Extract IK from Viser (highest impact)
1. Create `ik/solver.py` with pure `IKSolver` protocol
2. Create `ik/pink_solver.py` — extract algorithm from `yam_pink.py`
3. Create `ik/pyroki_solver.py` — extract algorithm from `yam_pyroki.py`
4. Create `visualization/ik_visualizer.py` — extract viz from `viser_base.py`
5. Update `YamViserAgent` to compose solver + visualizer
6. Update `YamVrAgent` to use pure solver (no viser_base import)
7. Delete `robots/viser/viser_base.py` (replaced by ik/ + visualization/)

### Phase B: Add observation schema
1. Create `core/observation.py` with `ObservationSpec` dataclass
2. Add `observation_spec()` to Agent protocol
3. Update `RobotEnv` to validate obs structure against spec
4. Update policy agents to use spec-driven obs mapping

### Phase C: Add data collection
1. Create `data/recorder.py` — hooks into `RobotEnv.step()`
2. Create `data/episode.py` — in-memory episode container
3. Add recording toggle to ViserMonitor (or standalone)
4. Support HDF5/zarr output for LeRobot compatibility

### Phase D: Clean up naming & structure
1. Rename files (drop redundant prefixes)
2. Move modules to new locations
3. Update all imports
4. Update YAML configs (_target_ paths)

---

## What NOT to Change

- `RobotEnv` control loop (works well at 100 Hz)
- Portal RPC architecture (needed for multi-process cameras/robots)
- YAML config system with `_target_` instantiation
- Camera driver interface (clean protocol)
- GELLO agent (already clean)
- YAM motor chain robot driver (hardware interface, don't touch)

---

## Decision Points for User

1. **Phase A first?** Extracting IK from viser is the biggest win. Do this
   before anything else.

2. **diffusion_policy_agent.py** — Delete now (rebuild when server exists) or
   fix imports to stub modules?

3. **Data format** — HDF5 (simple), Zarr (chunked), or LeRobot format
   (community standard)?

4. **Rename files now or later?** Renaming changes all `_target_` paths in
   YAML configs, so it's a bigger diff. Can defer.
