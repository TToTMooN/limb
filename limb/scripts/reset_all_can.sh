#!/bin/bash

BITRATE=1000000

if [ "$(id -u)" != "0" ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Check if a CAN interface is already up with the correct bitrate AND not wedged.
# A bus that's UP but in BUS-OFF / ERROR-PASSIVE state will silently drop frames;
# we need to bounce it in that case.
is_can_ok() {
    local iface=$1
    if ! ip link show "$iface" 2>/dev/null | grep -q "UP"; then
        return 1
    fi
    if ! ip -details link show "$iface" 2>/dev/null | grep -q "bitrate $BITRATE"; then
        return 1
    fi
    # ERROR-ACTIVE = healthy.  Anything else (ERROR-WARNING / ERROR-PASSIVE / BUS-OFF
    # / STOPPED) means the controller has been seeing TX/RX errors and needs a reset
    # to recover.  Match these states explicitly to avoid colliding with the netdev
    # "state UP" line that also appears in `ip` output.
    if ip -details link show "$iface" 2>/dev/null \
        | grep -qE "(BUS-OFF|ERROR-PASSIVE|ERROR-WARNING|STOPPED)"; then
        echo "CAN interface $iface is in a degraded controller state — will reset."
        return 1
    fi
    return 0
}

# Function to reset a CAN interface
reset_can_interface() {
    local iface=$1
    if is_can_ok "$iface"; then
        echo "CAN interface $iface already UP at ${BITRATE}bps — skipping."
        return 0
    fi
    echo "Resetting CAN interface: $iface"
    $SUDO ip link set "$iface" down
    $SUDO ip link set "$iface" up type can bitrate $BITRATE
}

# Get all CAN interfaces
can_interfaces=$(ip link show | grep -oP '(?<=: )(can\w+)')

# Check if any CAN interfaces were found
if [[ -z "$can_interfaces" ]]; then
    echo "No CAN interfaces found."
    exit 0
fi

# Reset each CAN interface only if needed
echo "Detected CAN interfaces: $can_interfaces"
for iface in $can_interfaces; do
    reset_can_interface "$iface"
done

echo "CAN setup complete."
