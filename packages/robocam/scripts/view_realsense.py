#!/usr/bin/env python3
"""Live viewer for Intel RealSense cameras.

Auto-detects all connected cameras and lets you choose which one(s) to stream.

Controls
--------
s  - toggle serial-number overlay
r  - start / stop video recording (uses AsyncVideoWriter + NVENC)
q  - quit (or ESC)

Examples
--------
    uv run scripts/view_realsense.py
    uv run scripts/view_realsense.py --serials 346123070863
    uv run scripts/view_realsense.py --all
    uv run scripts/view_realsense.py --list
    uv run scripts/view_realsense.py --depth
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tyro
from dataclasses import dataclass
from loguru import logger

from robocam import AsyncVideoWriter
from robocam.drivers.realsense import RealsenseCamera, discover_devices


DEFAULT_RESOLUTION: Dict[str, tuple[int, int]] = {
    "D455": (1280, 720),
    "D435": (1280, 720),
    "D415": (1280, 720),
    "D405": (640, 480),
}
DEFAULT_FPS = 30
FALLBACK_RESOLUTION = (640, 480)


def resolution_for_model(name: str) -> tuple[int, int]:
    for model_key, res in DEFAULT_RESOLUTION.items():
        if model_key in name:
            return res
    return FALLBACK_RESOLUTION


def pick_cameras(devices: List[Dict[str, str]]) -> List[str]:
    print(f"\nDetected {len(devices)} RealSense camera(s):\n")
    for i, d in enumerate(devices):
        print(f"  [{i}]  {d['serial']}  ({d['name']})")
    print("  [a]  All cameras")
    print()
    while True:
        choice = input("Select camera number (or 'a' for all): ").strip().lower()
        if choice == "a":
            return [d["serial"] for d in devices]
        try:
            idx = int(choice)
            if 0 <= idx < len(devices):
                return [devices[idx]["serial"]]
        except ValueError:
            pass
        print(f"  Invalid choice. Enter 0-{len(devices) - 1} or 'a'.")


def build_grid(tiles: List[np.ndarray], tile_w: int, tile_h: int) -> np.ndarray:
    n = len(tiles)
    if n == 0:
        return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        resized = cv2.resize(tile, (tile_w, tile_h))
        grid[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = resized
    return grid


def overlay_text(image: np.ndarray, text: str, position: tuple = (10, 30)) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    cv2.putText(image, text, (position[0] + 1, position[1] + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(image, text, position, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)


@dataclass
class Args:
    """RealSense camera live viewer & recorder."""

    list_devices: bool = False
    """List connected devices and exit."""
    serials: Optional[List[str]] = None
    """Specific serial number(s) to open."""
    all: bool = False
    """Open all cameras (skip interactive picker)."""
    fps: int = DEFAULT_FPS
    """Stream FPS."""
    show_serial: bool = False
    """Show serial overlay from the start."""
    depth: bool = False
    """Enable and show colorized depth stream."""
    flip_ud: bool = False
    """Flip image upside-down."""
    flip_lr: bool = False
    """Flip image left-right (mirror)."""
    output_dir: Path = Path("recordings")
    """Base dir for recordings."""


def main() -> None:
    args = tyro.cli(Args)

    all_devices = discover_devices()
    if args.list_devices:
        if not all_devices:
            print("No RealSense devices found.")
        else:
            print(f"Found {len(all_devices)} RealSense device(s):")
            for d in all_devices:
                print(f"  {d['serial']}  {d['name']}")
        return

    if not all_devices:
        print("No RealSense devices found. Exiting.")
        return

    serial_to_name = {d["serial"]: d["name"] for d in all_devices}

    if args.serials:
        serials = args.serials
    elif args.all:
        serials = [d["serial"] for d in all_devices]
    else:
        serials = pick_cameras(all_devices)

    # Open cameras using robocam driver
    cameras: Dict[str, RealsenseCamera] = {}
    for s in serials:
        cam_name = serial_to_name.get(s, "unknown")
        w, h = resolution_for_model(cam_name)
        logger.info("Opening {} ({}) at {}x{} @ {}fps", s, cam_name, w, h, args.fps)
        try:
            cameras[s] = RealsenseCamera(
                serial_number=s, resolution=(w, h), fps=args.fps, enable_depth=args.depth
            )
        except Exception as e:
            logger.warning("Failed to open {}: {}", s, e)

    if not cameras:
        print("No cameras could be opened. Exiting.")
        return

    show_serial = args.show_serial
    recording = False
    writers: Dict[str, AsyncVideoWriter] = {}
    record_dir: Optional[Path] = None

    window_name = "RealSense Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nControls:  [s] toggle serial overlay  |  [r] toggle recording  |  [q/ESC] quit\n")

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            tiles: List[np.ndarray] = []

            for serial, cam in cameras.items():
                data = cam.read()
                rgb = data.images.get("rgb")
                if rgb is None:
                    rgb = np.zeros((cam.resolution[1], cam.resolution[0], 3), dtype=np.uint8)

                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if args.flip_ud:
                    bgr = cv2.flip(bgr, 0)
                if args.flip_lr:
                    bgr = cv2.flip(bgr, 1)

                # FPS overlay
                frame_count += 1
                elapsed = time.time() - t0
                if elapsed > 0:
                    overlay_text(bgr, f"{frame_count / elapsed:.1f} fps", (bgr.shape[1] - 150, 30))

                if show_serial:
                    overlay_text(bgr, f"SN: {serial}")

                tiles.append(bgr)

                # Depth tile
                if args.depth and "depth" in data.images:
                    depth = data.images["depth"]
                    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                    if show_serial:
                        overlay_text(depth_color, f"SN: {serial} (depth)")
                    tiles.append(depth_color)

                # Recording
                if recording and serial in writers:
                    writers[serial].write(rgb)

            tile_h, tile_w = tiles[0].shape[:2] if tiles else (480, 640)
            grid = build_grid(tiles, tile_w, tile_h)

            if recording:
                overlay_text(grid, "REC", (grid.shape[1] - 80, 30))

            cv2.imshow(window_name, grid)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("s"):
                show_serial = not show_serial

            if key == ord("r"):
                if not recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    record_dir = args.output_dir / f"recording_{ts}"
                    record_dir.mkdir(parents=True, exist_ok=True)
                    for serial, cam in cameras.items():
                        w, h = cam.resolution
                        path = str(record_dir / f"cam_{serial}.mp4")
                        writer = AsyncVideoWriter(path=path, width=w, height=h, fps=args.fps)
                        writer.start()
                        writers[serial] = writer
                        logger.info("Recording -> {}", path)
                    recording = True
                    print("Recording started.")
                else:
                    for w in writers.values():
                        w.stop()
                    writers.clear()
                    recording = False
                    print(f"Recording stopped. Files saved to {record_dir}")

    except KeyboardInterrupt:
        pass
    finally:
        for w in writers.values():
            w.stop()
        if record_dir and writers:
            print(f"Recording saved to {record_dir}")

        for cam in cameras.values():
            cam.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
