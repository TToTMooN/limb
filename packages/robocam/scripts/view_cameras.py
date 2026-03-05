#!/usr/bin/env python3
"""Live viewer for all connected cameras (RealSense + ZED).

Auto-discovers all RealSense and ZED cameras and displays them in a grid.

Controls
--------
s  - toggle device-id overlay
r  - start / stop video recording (uses AsyncVideoWriter + NVENC)
q  - quit (or ESC)

Examples
--------
    uv run scripts/view_cameras.py
    uv run scripts/view_cameras.py --depth
    uv run scripts/view_cameras.py --list
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tyro
from loguru import logger

from robocam import AsyncVideoWriter
from robocam.camera import CameraData, CameraDriver


def discover_all() -> List[Dict]:
    """Discover all connected RealSense and ZED cameras."""
    found: List[Dict] = []

    # RealSense
    try:
        from robocam.drivers.realsense import RealsenseCamera, discover_devices

        for dev in discover_devices():
            found.append({
                "type": "realsense",
                "serial": dev["serial"],
                "name": dev["name"],
                "factory": lambda d=dev: RealsenseCamera(serial_number=d["serial"], fps=30, enable_depth=_args.depth),
            })
    except ImportError:
        logger.debug("pyrealsense2 not available, skipping RealSense discovery")

    # ZED
    try:
        from pyzed import sl

        for cam_info in sl.Camera.get_device_list():
            serial = str(cam_info.serial_number)
            found.append({
                "type": "zed",
                "serial": serial,
                "name": f"ZED ({cam_info.camera_model})",
                "factory": lambda s=serial: _open_zed(s),
            })
    except ImportError:
        logger.debug("pyzed not available, skipping ZED discovery")

    return found


# Global ref so lambdas in discover_all can read args
_args: Args = None  # type: ignore[assignment]


def _open_zed(serial: str) -> CameraDriver:
    from robocam.drivers.zed import ZedCamera

    return ZedCamera(device_id=serial, resolution=_args.zed_resolution, fps=_args.zed_fps, enable_depth=_args.depth)


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


def get_rgb(data: CameraData) -> Optional[np.ndarray]:
    """Extract the main RGB image from camera data."""
    for key in ("rgb", "left_rgb"):
        img = data.images.get(key)
        if img is not None:
            return img
    return None


@dataclass
class Args:
    """View all connected cameras (RealSense + ZED) in a single window."""

    list_devices: bool = False
    """List connected devices and exit."""
    depth: bool = False
    """Enable and show depth streams."""
    show_id: bool = False
    """Show device-id overlay from the start."""
    zed_resolution: str = "HD720"
    """ZED resolution: HD2K, HD1200, HD1080, HD720, VGA, SVGA."""
    zed_fps: int = 30
    """ZED target frame rate."""
    output_dir: Path = Path("recordings")
    """Base dir for recordings."""


def main() -> None:
    global _args
    _args = tyro.cli(Args)

    devices = discover_all()

    if _args.list_devices:
        if not devices:
            print("No cameras found.")
        else:
            print(f"Found {len(devices)} camera(s):")
            for d in devices:
                print(f"  [{d['type']}]  {d['serial']}  ({d['name']})")
        return

    if not devices:
        print("No cameras found. Exiting.")
        return

    print(f"\nFound {len(devices)} camera(s):")
    for d in devices:
        print(f"  [{d['type']}]  {d['serial']}  ({d['name']})")

    # Open all cameras
    cameras: Dict[str, CameraDriver] = {}
    camera_labels: Dict[str, str] = {}
    for d in devices:
        label = f"{d['type']}:{d['serial']}"
        logger.info("Opening {} ({})", label, d["name"])
        try:
            cameras[label] = d["factory"]()
            camera_labels[label] = f"{d['type'].upper()} {d['serial']}"
        except Exception as e:
            logger.warning("Failed to open {}: {}", label, e)

    if not cameras:
        print("No cameras could be opened. Exiting.")
        return

    show_overlay = _args.show_id
    recording = False
    writers: Dict[str, AsyncVideoWriter] = {}
    record_dir: Optional[Path] = None

    window_name = "All Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nControls:  [s] toggle id overlay  |  [r] toggle recording  |  [q/ESC] quit\n")

    cam_frame_counts: Dict[str, int] = {label: 0 for label in cameras}
    t0 = time.time()

    try:
        while True:
            tiles: List[np.ndarray] = []
            frame_rgbs: Dict[str, np.ndarray] = {}
            elapsed = time.time() - t0

            for label, cam in cameras.items():
                data = cam.read()
                rgb = get_rgb(data)
                if rgb is None:
                    continue

                cam_frame_counts[label] += 1
                frame_rgbs[label] = rgb
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if show_overlay:
                    if elapsed > 0:
                        fps = cam_frame_counts[label] / elapsed
                        overlay_text(bgr, f"{fps:.1f} fps", (bgr.shape[1] - 150, 30))
                    overlay_text(bgr, camera_labels[label])

                tiles.append(bgr)

            if not tiles:
                continue

            tile_h, tile_w = tiles[0].shape[:2]
            grid = build_grid(tiles, tile_w, tile_h)

            if recording:
                overlay_text(grid, "REC", (grid.shape[1] - 80, 30))

            cv2.imshow(window_name, grid)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                show_overlay = not show_overlay
            if key == ord("r"):
                if not recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    record_dir = _args.output_dir / f"all_cameras_{ts}"
                    record_dir.mkdir(parents=True, exist_ok=True)
                    for label, rgb in frame_rgbs.items():
                        h, w = rgb.shape[:2]
                        safe_label = label.replace(":", "_")
                        path = str(record_dir / f"{safe_label}.mp4")
                        writer = AsyncVideoWriter(path=path, width=w, height=h, fps=30)
                        writer.start()
                        writers[label] = writer
                        logger.info("Recording -> {}", path)
                    recording = True
                else:
                    for w in writers.values():
                        w.stop()
                    writers.clear()
                    recording = False
                    logger.info("Recording stopped -> {}", record_dir)

            # Write frames
            if recording:
                for label, rgb in frame_rgbs.items():
                    if label in writers:
                        writers[label].write(rgb)

    except KeyboardInterrupt:
        pass
    finally:
        for w in writers.values():
            w.stop()
        for cam in cameras.values():
            cam.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
