"""Post-hoc episode success marking.

Mark recorded episodes as successful after the fact so they can be filtered
at conversion time with `limb convert-lerobot --success-only`. Useful when
the collection pedal doesn't mark success inline — record everything, review
later, commit the good ones.

Usage:
    uv run limb mark --session-dir recordings/task                  # interactive
    uv run limb mark --session-dir recordings/task --all            # mark all
    uv run limb mark --session-dir recordings/task --clear          # unmark all
    uv run limb mark --episode-dir recordings/task/episode_...      # single
    uv run limb mark --episode-dir recordings/task/episode_... --clear
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger


@dataclass
class Args:
    session_dir: Optional[str] = None
    episode_dir: Optional[str] = None
    all: bool = False
    clear: bool = False


def _find_session_episodes(session_dir: Path) -> List[Path]:
    if not session_dir.exists():
        logger.error("Session directory not found: {}", session_dir)
        raise SystemExit(1)
    episodes = sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    valid: List[Path] = []
    for ep in episodes:
        if (ep / "RECORDING_IN_PROGRESS").exists():
            continue
        if not (ep / "metadata.json").exists():
            continue
        valid.append(ep)
    return valid


def _summary(episode_dir: Path) -> str:
    """One-line summary of an episode for interactive review."""
    meta_path = episode_dir / "metadata.json"
    if not meta_path.exists():
        return f"{episode_dir.name} (no metadata)"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        return f"{episode_dir.name} (unreadable metadata)"
    num_steps = meta.get("num_steps", "?")
    duration = meta.get("duration_s", 0.0)
    marked = "SUCCESS" if (episode_dir / "SUCCESS").exists() else "unmarked"
    return f"{episode_dir.name}  [{num_steps} steps, {duration:.1f}s, {marked}]"


def _mark(episode_dir: Path, success: bool) -> bool:
    """Create or delete the SUCCESS marker. Returns True if state changed."""
    marker = episode_dir / "SUCCESS"
    if success:
        if marker.exists():
            return False
        marker.touch()
        return True
    if not marker.exists():
        return False
    marker.unlink()
    return True


def _mark_all(episodes: List[Path], success: bool) -> None:
    verb = "Marking" if success else "Unmarking"
    logger.info("{} {} episode(s) in batch...", verb, len(episodes))
    changed = 0
    for ep in episodes:
        if _mark(ep, success):
            changed += 1
    logger.info("Done. {} episode(s) changed, {} already in desired state.", changed, len(episodes) - changed)


def _interactive(episodes: List[Path]) -> None:
    """Walk episodes one at a time, prompting the user for a decision."""
    logger.info("Interactive review: {} episode(s)", len(episodes))
    logger.info("Controls: [y] mark success  [n] unmark  [s] skip  [q] quit")

    marked = 0
    unmarked = 0
    skipped = 0
    for idx, ep in enumerate(episodes, start=1):
        while True:
            prompt = f"[{idx}/{len(episodes)}] {_summary(ep)} > "
            try:
                choice = input(prompt).strip().lower()
            except EOFError:
                logger.info("EOF — ending review")
                return
            if choice in ("y", "yes"):
                if _mark(ep, True):
                    marked += 1
                break
            if choice in ("n", "no"):
                if _mark(ep, False):
                    unmarked += 1
                break
            if choice in ("s", "skip", ""):
                skipped += 1
                break
            if choice in ("q", "quit"):
                logger.info("Review stopped early. marked={}, unmarked={}, skipped={}", marked, unmarked, skipped)
                return
            print("  y=mark success, n=unmark, s=skip, q=quit")
    logger.info("Review complete. marked={}, unmarked={}, skipped={}", marked, unmarked, skipped)


def main(args: Args) -> None:
    if not args.session_dir and not args.episode_dir:
        logger.error("Must provide --session-dir or --episode-dir")
        raise SystemExit(1)
    if args.session_dir and args.episode_dir:
        logger.error("Pass exactly one of --session-dir or --episode-dir")
        raise SystemExit(1)

    # Single-episode mode: always non-interactive (success by default, or clear)
    if args.episode_dir:
        ep = Path(args.episode_dir)
        if not ep.exists():
            logger.error("Episode directory not found: {}", ep)
            raise SystemExit(1)
        success = not args.clear
        changed = _mark(ep, success)
        state = "SUCCESS" if success else "unmarked"
        if changed:
            logger.info("{} -> {}", ep.name, state)
        else:
            logger.info("{} already {} — no change", ep.name, state)
        return

    # Session mode
    session_dir = Path(args.session_dir)  # type: ignore[arg-type]
    episodes = _find_session_episodes(session_dir)
    if not episodes:
        logger.error("No valid episodes found in {}", session_dir)
        raise SystemExit(1)

    if args.clear:
        _mark_all(episodes, success=False)
        return
    if args.all:
        _mark_all(episodes, success=True)
        return
    _interactive(episodes)
