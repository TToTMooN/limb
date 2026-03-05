#!/usr/bin/env python3
"""Live viewer for Stereolabs ZED cameras.

Controls
--------
s  - toggle device-id overlay
r  - start / stop video recording (uses AsyncVideoWriter + NVENC)
q  - quit (or ESC)

Examples
--------
    uv run scripts/view_zed.py
    uv run scripts/view_zed.py --device-id 12345678
    uv run scripts/view_zed.py --resolution HD720 --fps 30
    uv run scripts/view_zed.py --depth
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro
from loguru import logger

from robocam import AsyncVideoWriter
from robocam.drivers.zed import ZedCamera


def overlay_text(image: np.ndarray, text: str, position: tuple = (10, 30)) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    cv2.putText(image, text, (position[0] + 1, position[1] + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(image, text, position, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)


@dataclass
class Args:
    """ZED camera live viewer & recorder."""

    device_id: Optional[str] = None
    """ZED serial number. None = first available."""
    resolution: str = "HD720"
    """Resolution: HD2K, HD1200, HD1080, HD720, VGA, SVGA."""
    fps: int = 30
    """Target frame rate."""
    depth: bool = False
    """Enable and show neural depth."""
    stereo: bool = False
    """Show right image alongside left."""
    show_id: bool = False
    """Show device-id overlay from the start."""
    output_dir: Path = Path("recordings")
    """Base dir for recordings."""


def main() -> None:
    args = tyro.cli(Args)

    logger.info("Opening ZED camera (device_id={}, {}@{}fps)", args.device_id, args.resolution, args.fps)
    cam = ZedCamera(
        device_id=args.device_id,
        resolution=args.resolution,
        fps=args.fps,
        enable_depth=args.depth,
        return_right_image=args.stereo,
    )

    show_id = args.show_id
    recording = False
    writer: Optional[AsyncVideoWriter] = None
    record_dir: Optional[Path] = None

    window_name = "ZED Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nControls:  [s] toggle id overlay  |  [r] toggle recording  |  [q/ESC] quit\n")

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            data = cam.read()
            if data.timestamp < 0:
                continue  # grab failed

            # Build display image
            left = data.images.get("left_rgb")
            if left is None:
                left = data.images.get("rgb")
            if left is None:
                continue

            bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
            panels = [bgr]

            if args.stereo and "right_rgb" in data.images:
                right_bgr = cv2.cvtColor(data.images["right_rgb"], cv2.COLOR_RGB2BGR)
                panels.append(right_bgr)

            if args.depth and hasattr(data, "depth_data") and data.depth_data is not None:
                depth_norm = cv2.normalize(data.depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                panels.append(depth_color)

            display = np.hstack(panels) if len(panels) > 1 else panels[0]

            # Overlays
            frame_count += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                overlay_text(display, f"{frame_count / elapsed:.1f} fps", (display.shape[1] - 150, 30))
            if show_id:
                overlay_text(display, f"ID: {cam.device_id or cam.serial_number}")
            if recording:
                overlay_text(display, "REC", (display.shape[1] - 80, 60))

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                show_id = not show_id
            if key == ord("r"):
                if not recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    record_dir = args.output_dir / f"zed_recording_{ts}"
                    record_dir.mkdir(parents=True, exist_ok=True)
                    h, w = left.shape[:2]
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

            # Write frame
            if recording and writer:
                writer.write(left)

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
