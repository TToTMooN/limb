"""Convert limb raw recordings to LeRobot v2.1 dataset format.

Thin wrapper — core logic lives in limb.data.convert_lerobot.

Usage:
    uv run scripts/data/convert_to_lerobot.py --input-dir recordings/task --output-dir datasets/task
    uv run limb convert-lerobot --input-dir recordings/task --output-dir datasets/task
"""

import tyro

from limb.data.convert_lerobot import Args, main

if __name__ == "__main__":
    main(tyro.cli(Args))
