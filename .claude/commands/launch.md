Launch the robot system with a given config.

Usage: /launch [config_name]

Available configs:
- viser-bimanual (default): configs/yam_viser_bimanual.yaml
- viser-single: configs/yam_viser_single_arm.yaml
- gello: configs/yam_gello_bimanual.yaml
- vr: configs/yam_vr_bimanual.yaml

Run the launch command:
```bash
uv run limb/envs/launch.py --config_path configs/yam_viser_bimanual.yaml
```

If the user specifies a config name, map it to the appropriate path above.
If the user specifies a full config path, use it directly.
Always use `uv run` prefix.
