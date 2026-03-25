"""Visualize a recorded episode using Rerun.

Thin wrapper — core logic lives in limb.data.visualize_episode.

Usage:
    uv run scripts/data/visualize_episode.py --episode-dir recordings/episode_...
    uv run limb visualize --episode-dir recordings/episode_...
"""

import tyro

from limb.data.visualize_episode import Args, main

if __name__ == "__main__":
    main(tyro.cli(Args))
