#!/usr/bin/env python3
"""Stream depth from the two RealSense D405 wrist cameras.

Uses limb's :class:`RealsenseCamera` driver with ``enable_depth=True`` (depth
aligned to color). Both cameras are polled **sequentially on the main thread** —
this is a hard pyrealsense2 constraint (background-thread polling stalls after
~16 frames).

Shows a live side-by-side window: color on top, colorized depth below, per arm.
The center-pixel depth (in metres) is printed on each tile.

Examples
--------
    # Default: left + right wrist serials from yam_gello_bimanual.yaml
    uv run scripts/diagnostics/test_wrist_depth.py

    # Explicit serials (left first, then right)
    uv run scripts/diagnostics/test_wrist_depth.py --serials 409122274017 409122274543

    # Auto-detect every connected D405
    uv run scripts/diagnostics/test_wrist_depth.py --all

Keyboard: q / ESC to quit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import tyro
from loguru import logger
from robocam.drivers.realsense import RealsenseCamera, discover_devices

# Wrist camera serials from configs/yam_gello_bimanual.yaml.
DEFAULT_WRIST_SERIALS: dict[str, str] = {
    "left_wrist": "409122274017",
    "right_wrist": "409122274543",
}
# D405 depth scale is 0.1 mm/unit, i.e. raw z16 * 0.0001 = metres.
D405_DEPTH_SCALE_M = 0.0001


@dataclass
class Args:
    serials: list[str] = field(default_factory=list)
    """Camera serials to open (order = tile order). Empty → wrist defaults."""
    all: bool = False
    """Open every connected D405 instead of the wrist defaults."""
    width: int = 640
    height: int = 480
    fps: int = 30


def _resolve_serials(args: Args) -> list[tuple[str, str]]:
    """Return ``[(label, serial), ...]`` to open."""
    if args.all:
        devs = [d for d in discover_devices() if "D405" in d["name"]]
        if not devs:
            raise SystemExit("No D405 cameras detected. Try `uv run limb devices`.")
        return [(d["serial"], d["serial"]) for d in devs]

    serials = args.serials or list(DEFAULT_WRIST_SERIALS.values())
    labels = list(DEFAULT_WRIST_SERIALS) if not args.serials else serials
    return list(zip(labels, serials, strict=True))


def _tile(label: str, color_rgb: np.ndarray, depth_raw: np.ndarray) -> np.ndarray:
    """Build a labelled color-over-depth tile (BGR, for cv2 display)."""
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

    depth_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET
    )
    if depth_vis.shape[:2] != color_bgr.shape[:2]:
        depth_vis = cv2.resize(depth_vis, (color_bgr.shape[1], color_bgr.shape[0]))

    h, w = depth_raw.shape[:2]
    center_m = float(depth_raw[h // 2, w // 2]) * D405_DEPTH_SCALE_M
    cv2.putText(color_bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(
        depth_vis,
        f"center: {center_m:.3f} m",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return np.vstack([color_bgr, depth_vis])


def main(args: Args) -> None:
    targets = _resolve_serials(args)
    logger.info("Opening {} camera(s): {}", len(targets), [s for _, s in targets])

    cams: list[tuple[str, RealsenseCamera]] = []
    for label, serial in targets:
        cam = RealsenseCamera(
            serial_number=serial,
            resolution=(args.width, args.height),
            fps=args.fps,
            enable_depth=True,
            name=label,
        )
        cams.append((label, cam))

    window = "wrist depth (q to quit)"
    try:
        while True:
            tiles = []
            for label, cam in cams:  # sequential main-thread poll — required for RealSense
                data = cam.read()
                depth = data.images.get("depth")
                if depth is None:
                    logger.warning("{}: no depth frame", label)
                    continue
                tiles.append(_tile(label, data.images["rgb"], depth))

            if tiles:
                grid = np.hstack(tiles)
                grid = cv2.resize(grid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imshow(window, grid)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for _, cam in cams:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(Args))
