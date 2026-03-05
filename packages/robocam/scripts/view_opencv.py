#!/usr/bin/env python3
"""Live viewer for generic V4L2 / OpenCV cameras.

Controls
--------
s  - toggle device-path overlay
r  - start / stop video recording (uses AsyncVideoWriter + NVENC)
q  - quit (or ESC)

Examples
--------
    uv run scripts/view_opencv.py --device-path /dev/video0
    uv run scripts/view_opencv.py --serial SN12345678
    uv run scripts/view_opencv.py --device-path /dev/video0 --resolution 1280 720
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tyro
from loguru import logger

from robocam import AsyncVideoWriter
from robocam.drivers.opencv import OpencvCamera


def overlay_text(image: np.ndarray, text: str, position: tuple = (10, 30)) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    cv2.putText(image, text, (position[0] + 1, position[1] + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(image, text, position, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)


@dataclass
class Args:
    """OpenCV camera live viewer & recorder."""

    device_path: str = ""
    """e.g. /dev/video0"""
    serial: Optional[str] = None
    """Serial number (resolves via /dev/v4l/by-id/)."""
    resolution: Tuple[int, int] = (640, 480)
    """(width, height)."""
    fps: int = 30
    """Target frame rate."""
    show_path: bool = False
    """Show device-path overlay from the start."""
    output_dir: Path = Path("recordings")
    """Base dir for recordings."""


def main() -> None:
    args = tyro.cli(Args)

    if not args.device_path and not args.serial:
        print("Provide --device-path or --serial. Example: --device-path /dev/video0")
        return

    logger.info("Opening camera (device={}, serial={})", args.device_path, args.serial)
    cam = OpencvCamera(
        device_path=args.device_path,
        serial_number=args.serial,
        resolution=args.resolution,
        fps=args.fps,
    )

    show_path = args.show_path
    recording = False
    writer: Optional[AsyncVideoWriter] = None
    record_dir: Optional[Path] = None

    window_name = "OpenCV Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nControls:  [s] toggle path overlay  |  [r] toggle recording  |  [q/ESC] quit\n")

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            data = cam.read()
            rgb = data.images["rgb"]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            frame_count += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                overlay_text(bgr, f"{frame_count / elapsed:.1f} fps", (bgr.shape[1] - 150, 30))
            if show_path:
                overlay_text(bgr, f"dev: {cam.device_path}")
            if recording:
                overlay_text(bgr, "REC", (bgr.shape[1] - 80, 60))

            cv2.imshow(window_name, bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                show_path = not show_path
            if key == ord("r"):
                if not recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    record_dir = args.output_dir / f"opencv_recording_{ts}"
                    record_dir.mkdir(parents=True, exist_ok=True)
                    w, h = args.resolution
                    path = str(record_dir / "video.mp4")
                    writer = AsyncVideoWriter(path=path, width=w, height=h, fps=args.fps)
                    writer.start()
                    recording = True
                    logger.info("Recording -> {}", path)
                else:
                    if writer:
                        writer.stop()
                        writer = None
                    recording = False
                    logger.info("Recording stopped -> {}", record_dir)

            if recording and writer:
                writer.write(rgb)

    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.stop()
        cam.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
