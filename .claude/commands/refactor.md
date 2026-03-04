Execute the YAM-only refactor: remove Franka and sim code.

Read .claude/refactor-plan.md for the complete checklist.

Before deleting anything:
1. Confirm with the user which phase to execute (Franka cleanup, sim cleanup, or both)
2. Show the list of files that will be deleted
3. Ask for confirmation before proceeding

Execution steps per phase:

## Phase 1: Franka cleanup
Delete files listed under "Franka robot driver & support" in refactor-plan.md.
Then remove franka entries from pyproject.toml.

## Phase 2: Sim cleanup
Delete files listed under "Sim-only code" and "Sim-only agents" in refactor-plan.md.

## Phase 3: Verify
```bash
uv run ruff check limb/
```
Fix any broken imports found.

## Phase 4: Smoke test
```bash
uv run limb/envs/launch.py --config_path configs/yam_viser_bimanual.yaml
```
(Will fail if real hardware not connected, but import errors will surface here.)

Update the checkboxes in .claude/refactor-plan.md as items are completed.
