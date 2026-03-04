#!/usr/bin/env python3
"""Diagnostic tool for RealSense cameras.

Auto-detects all connected RealSense cameras and lets you choose which
one(s) to stream in a live OpenCV window.

Keyboard controls
-----------------
s  - toggle serial-number overlay on each tile
r  - start / stop video recording
q  - quit (or ESC)

Examples
--------
    # Interactive picker (default)
    python scripts/test_realsense_cameras.py

    # Open a specific camera by serial
    python scripts/test_realsense_cameras.py --serials 346123070863

    # Open all cameras (skip the picker)
    python scripts/test_realsense_cameras.py --all

    # List connected devices and exit
    python scripts/test_realsense_cameras.py --list
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as _e:
    raise ImportError(
        "pyrealsense2 is required for this script. "
        "Install it with:  pip install pyrealsense2"
    ) from _e


DEFAULT_RESOLUTION: Dict[str, tuple[int, int]] = {
    "D455": (1280, 720),
    "D435": (1280, 720),
    "D415": (1280, 720),
    "D405": (640, 480),
}
DEFAULT_FPS = 30
FALLBACK_RESOLUTION = (640, 480)


def discover_devices() -> List[Dict[str, str]]:
    ctx = rs.context()
    return [
        {
            "serial": d.get_info(rs.camera_info.serial_number),
            "name": d.get_info(rs.camera_info.name),
        }
        for d in ctx.query_devices()
    ]


def resolution_for_model(name: str) -> tuple[int, int]:
    """Return default (w, h) based on camera model name."""
    for model_key, res in DEFAULT_RESOLUTION.items():
        if model_key in name:
            return res
    return FALLBACK_RESOLUTION


def pick_cameras(devices: List[Dict[str, str]]) -> List[str]:
    """Interactive menu: print detected cameras and let the user choose."""
    print(f"\nDetected {len(devices)} RealSense camera(s):\n")
    for i, d in enumerate(devices):
        print(f"  [{i}]  {d['serial']}  ({d['name']})")
    print(f"  [a]  All cameras")
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


FRAME_TIMEOUT_MS = 3000
WARMUP_FRAMES = 3


class CameraPipeline:
    """Thin wrapper around a single RealSense pipeline for the test viewer."""

    def __init__(self, serial: str, width: int, height: int, fps: int, enable_depth: bool) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.enable_depth = enable_depth

        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color) if enable_depth else None
        self.colorizer = rs.colorizer() if enable_depth else None

        print(f"  Warming up ({WARMUP_FRAMES} frames) ...", end="", flush=True)
        ok = 0
        for _ in range(WARMUP_FRAMES):
            try:
                self.pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
                print(".", end="", flush=True)
                ok += 1
            except RuntimeError:
                print("x", end="", flush=True)
        if ok > 0:
            print(" ready.")
        else:
            print(f" FAILED - no frames at {width}x{height}@{fps}fps.")
            self.pipeline.stop()
            raise RuntimeError(
                f"Camera {serial} delivered 0/{WARMUP_FRAMES} warmup frames. "
                f"It may not support {width}x{height}@{fps} or could be faulty."
            )

    def read(self) -> Optional[Dict[str, np.ndarray]]:
        """Return ``{"color": ..., "depth_colorized": ...}`` numpy arrays (BGR), or None on timeout."""
        try:
            frames = self.pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        except RuntimeError:
            return None

        if self.align is not None:
            frames = self.align.process(frames)

        result: Dict[str, np.ndarray] = {}
        color_frame = frames.get_color_frame()
        if color_frame:
            color_bgr = cv2.cvtColor(np.asarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)
            result["color"] = color_bgr

        if self.enable_depth and self.colorizer is not None:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                colorized = np.asarray(self.colorizer.colorize(depth_frame).get_data())
                result["depth_colorized"] = colorized

        return result

    def stop(self) -> None:
        self.pipeline.stop()


def build_grid(
    tiles: List[np.ndarray],
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    """Arrange *tiles* into a rectangular grid, padding with black if needed."""
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
    """Draw *text* with a dark shadow for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    cv2.putText(image, text, (position[0] + 1, position[1] + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(image, text, position, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RealSense camera live viewer & recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--list", action="store_true", help="List connected devices and exit")
    cam_group = p.add_mutually_exclusive_group()
    cam_group.add_argument("--serials", nargs="+", default=None, help="Specific serial number(s) to open")
    cam_group.add_argument("--all", action="store_true", help="Open all cameras (skip interactive picker)")
    p.add_argument("--width", type=int, default=None, help="Stream width (auto: max for single cam, 640 for multi)")
    p.add_argument("--height", type=int, default=None, help="Stream height (auto: max for single cam, 480 for multi)")
    p.add_argument("--fps", type=int, default=None, help="Stream FPS (auto: max for single cam, 30 for multi)")
    p.add_argument("--show-serial", action="store_true", help="Show serial overlay from the start")
    p.add_argument("--depth", action="store_true", help="Enable and show colorized depth stream")
    p.add_argument("--flip-ud", action="store_true", help="Flip image upside-down")
    p.add_argument("--flip-lr", action="store_true", help="Flip image left-right (mirror)")
    p.add_argument("--output-dir", type=Path, default=Path("recordings"), help="Base dir for recordings")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_devices = discover_devices()
    if args.list:
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

    fps = args.fps or DEFAULT_FPS

    pipelines: List[CameraPipeline] = []
    for s in serials:
        cam_name = serial_to_name.get(s, "unknown")
        if args.width is not None and args.height is not None:
            w, h = args.width, args.height
        else:
            w, h = resolution_for_model(cam_name)
        print(f"Opening {s} ({cam_name}) at {w}x{h} @ {fps}fps ...")
        try:
            pipelines.append(CameraPipeline(s, w, h, fps, args.depth))
        except RuntimeError as e:
            print(f"  WARNING: Failed to open {s}: {e}")
            print(f"  Try lowering --fps or --width/--height if USB bandwidth is limited.")

    if not pipelines:
        print("No cameras could be opened. Exiting.")
        return

    tile_w = max(p.width for p in pipelines)
    tile_h = max(p.height for p in pipelines)

    show_serial = args.show_serial
    recording = False
    writers: Dict[str, cv2.VideoWriter] = {}
    record_dir: Optional[Path] = None

    window_name = "RealSense Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nControls:  [s] toggle serial overlay  |  [r] toggle recording  |  [q/ESC] quit\n")

    try:
        while True:
            tiles: List[np.ndarray] = []

            for pipe in pipelines:
                images = pipe.read()
                if images is None:
                    images = {}
                color = images.get("color")
                if color is None:
                    color = np.zeros((pipe.height, pipe.width, 3), dtype=np.uint8)

                if args.flip_ud:
                    color = cv2.flip(color, 0)
                if args.flip_lr:
                    color = cv2.flip(color, 1)

                if show_serial:
                    overlay_text(color, f"SN: {pipe.serial}")

                tiles.append(color)

                if args.depth:
                    depth_vis = images.get("depth_colorized")
                    if depth_vis is None:
                        depth_vis = np.zeros_like(color)
                    if args.flip_ud:
                        depth_vis = cv2.flip(depth_vis, 0)
                    if args.flip_lr:
                        depth_vis = cv2.flip(depth_vis, 1)
                    if show_serial:
                        overlay_text(depth_vis, f"SN: {pipe.serial} (depth)")
                    tiles.append(depth_vis)

                if recording and pipe.serial in writers:
                    writers[pipe.serial].write(color)

            grid = build_grid(tiles, tile_w, tile_h)

            if recording:
                overlay_text(grid, "REC", (grid.shape[1] - 80, 30))

            cv2.imshow(window_name, grid)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("s"):
                show_serial = not show_serial
                state = "ON" if show_serial else "OFF"
                print(f"Serial overlay: {state}")

            if key == ord("r"):
                if not recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    record_dir = args.output_dir / f"recording_{ts}"
                    record_dir.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    for pipe in pipelines:
                        path = str(record_dir / f"cam_{pipe.serial}.mp4")
                        writers[pipe.serial] = cv2.VideoWriter(
                            path, fourcc, fps, (pipe.width, pipe.height)
                        )
                        print(f"  Recording -> {path}")
                    recording = True
                    print("Recording started.")
                else:
                    for w in writers.values():
                        w.release()
                    writers.clear()
                    recording = False
                    print(f"Recording stopped. Files saved to {record_dir}")

    except KeyboardInterrupt:
        pass
    finally:
        if writers:
            for w in writers.values():
                w.release()
            print(f"Recording saved to {record_dir}")

        for pipe in pipelines:
            pipe.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
