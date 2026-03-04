#!/bin/bash

BITRATE=1000000

if [ "$(id -u)" != "0" ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Check if a CAN interface is already up with the correct bitrate
is_can_ok() {
    local iface=$1
    # Interface must be UP and have the right bitrate
    if ip link show "$iface" 2>/dev/null | grep -q "UP"; then
        if ip -details link show "$iface" 2>/dev/null | grep -q "bitrate $BITRATE"; then
            return 0
        fi
    fi
    return 1
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
