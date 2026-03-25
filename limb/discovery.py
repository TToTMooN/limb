"""Hardware device discovery for limb.

Enumerates connected cameras, robot CAN interfaces, and input devices.
Inspired by Raiden's `rd list_devices` command.

Usage:
    uv run limb devices
    uv run limb devices --verbose
"""

from __future__ import annotations

from loguru import logger


def _discover_realsense() -> list[dict]:
    """Find connected Intel RealSense cameras."""
    results = []
    try:
        import pyrealsense2 as rs

        ctx = rs.context()
        for dev in ctx.query_devices():
            info = {
                "type": "realsense",
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
                "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor)
                if dev.supports(rs.camera_info.usb_type_descriptor)
                else "unknown",
            }
            results.append(info)
    except ImportError:
        logger.debug("pyrealsense2 not available, skipping RealSense discovery")
    except Exception as e:
        logger.debug("RealSense discovery error: {}", e)
    return results


def _discover_zed() -> list[dict]:
    """Find connected ZED cameras."""
    results = []
    try:
        from pyzed import sl

        devs = sl.Camera.get_device_list()
        for dev in devs:
            info = {
                "type": "zed",
                "model": str(dev.camera_model),
                "serial": str(dev.serial_number),
                "state": str(dev.camera_state),
            }
            results.append(info)
    except ImportError:
        logger.debug("pyzed not available, skipping ZED discovery")
    except Exception as e:
        logger.debug("ZED discovery error: {}", e)
    return results


def _discover_can_interfaces() -> list[dict]:
    """Find CAN network interfaces (for YAM motor chains)."""
    results = []
    try:
        import os

        net_dir = "/sys/class/net"
        if os.path.isdir(net_dir):
            for iface in sorted(os.listdir(net_dir)):
                if iface.startswith("can") or iface.startswith("vcan"):
                    state_file = os.path.join(net_dir, iface, "operstate")
                    state = "unknown"
                    try:
                        with open(state_file) as f:
                            state = f.read().strip()
                    except OSError:
                        pass
                    results.append({"type": "can", "interface": iface, "state": state})
    except Exception as e:
        logger.debug("CAN discovery error: {}", e)
    return results


def _discover_dynamixel() -> list[dict]:
    """Find Dynamixel USB serial devices (GELLO controllers)."""
    results = []
    try:
        import glob

        # Common Dynamixel USB-serial paths
        patterns = ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/serial/by-id/*dynamixel*", "/dev/serial/by-id/*FTDI*"]
        seen = set()
        for pattern in patterns:
            for path in sorted(glob.glob(pattern)):
                if path not in seen:
                    seen.add(path)
                    results.append({"type": "dynamixel_serial", "path": path})
    except Exception as e:
        logger.debug("Dynamixel discovery error: {}", e)
    return results


def _discover_foot_pedals() -> list[dict]:
    """Find USB foot pedals via evdev."""
    results = []
    try:
        import evdev

        KNOWN_PEDALS = {
            (0x3553, 0xB001): "PCsensor FootSwitch",
            (0x1A86, 0xE026): "iKKEGOL FootSwitch",
        }
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                key = (dev.info.vendor, dev.info.product)
                if key in KNOWN_PEDALS:
                    results.append(
                        {
                            "type": "foot_pedal",
                            "name": KNOWN_PEDALS[key],
                            "path": path,
                            "vendor": f"0x{dev.info.vendor:04x}",
                            "product": f"0x{dev.info.product:04x}",
                        }
                    )
                dev.close()
            except (OSError, PermissionError):
                continue
    except ImportError:
        logger.debug("evdev not available, skipping foot pedal discovery")
    except Exception as e:
        logger.debug("Foot pedal discovery error: {}", e)
    return results


def _discover_spacemouse() -> list[dict]:
    """Find 3Dconnexion SpaceMouse devices via evdev."""
    results = []
    try:
        import evdev

        SPACEMOUSE_VENDORS = {0x256F}  # 3Dconnexion
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                if dev.info.vendor in SPACEMOUSE_VENDORS:
                    results.append(
                        {
                            "type": "spacemouse",
                            "name": dev.name,
                            "path": path,
                            "vendor": f"0x{dev.info.vendor:04x}",
                            "product": f"0x{dev.info.product:04x}",
                        }
                    )
                dev.close()
            except (OSError, PermissionError):
                continue
    except ImportError:
        logger.debug("evdev not available, skipping SpaceMouse discovery")
    except Exception as e:
        logger.debug("SpaceMouse discovery error: {}", e)
    return results


def discover_devices(verbose: bool = False) -> dict[str, list[dict]]:
    """Discover all connected hardware and print a summary.

    Returns a dict of device category -> list of device info dicts.
    """
    logger.info("Scanning for connected devices...\n")

    all_devices: dict[str, list[dict]] = {}

    # Cameras
    realsense = _discover_realsense()
    zed = _discover_zed()
    cameras = realsense + zed
    all_devices["cameras"] = cameras

    if cameras:
        logger.info("Cameras ({}):", len(cameras))
        for cam in cameras:
            if cam["type"] == "realsense":
                usb = f" (USB {cam['usb_type']})" if verbose else ""
                logger.info("  [RealSense] {} — serial: {}{}", cam["name"], cam["serial"], usb)
            elif cam["type"] == "zed":
                logger.info("  [ZED] {} — serial: {}", cam["model"], cam["serial"])
    else:
        logger.info("Cameras: none found")

    # CAN interfaces (robot arms)
    can_ifaces = _discover_can_interfaces()
    all_devices["can_interfaces"] = can_ifaces

    if can_ifaces:
        logger.info("\nCAN interfaces ({}):", len(can_ifaces))
        for iface in can_ifaces:
            logger.info("  {} — state: {}", iface["interface"], iface["state"])
    else:
        logger.info("\nCAN interfaces: none found")

    # Dynamixel serial (GELLO)
    dxl = _discover_dynamixel()
    all_devices["dynamixel"] = dxl

    if dxl:
        logger.info("\nDynamixel serial ports ({}):", len(dxl))
        for d in dxl:
            logger.info("  {}", d["path"])
    else:
        logger.info("\nDynamixel serial ports: none found")

    # Input devices
    pedals = _discover_foot_pedals()
    spacemice = _discover_spacemouse()
    inputs = pedals + spacemice
    all_devices["input_devices"] = inputs

    if inputs:
        logger.info("\nInput devices ({}):", len(inputs))
        for dev in inputs:
            extra = f" — {dev['path']}" if verbose else ""
            logger.info("  [{}] {}{}", dev["type"], dev.get("name", "unknown"), extra)
    else:
        logger.info("\nInput devices: none found (foot pedals, SpaceMouse)")

    total = sum(len(v) for v in all_devices.values())
    logger.info("\nTotal: {} device(s) found", total)

    return all_devices
