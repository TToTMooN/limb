#!/bin/bash
# Launch the GELLO position server on the R1 Lite Teleop onboard computer.
#
# Usage:
#   bash scripts/start_gello_server.sh            # default host
#   bash scripts/start_gello_server.sh 10.42.0.2  # custom host
#   bash scripts/start_gello_server.sh --kill      # just kill, don't restart
#
# What it does:
#   1. Kills any existing gello_position_server.py on the remote
#   2. Copies the latest script over
#   3. Starts it in a detached screen session
#
# Prerequisites:
#   - SSH key auth set up for the remote (ssh-copy-id cat@10.42.0.1)
#   - screen installed on the remote (sudo apt install screen)

REMOTE_USER="cat"
REMOTE_SCRIPT="gello_position_server.py"
LOCAL_SCRIPT="scripts/gello_position_server.py"
SCREEN_NAME="gello_server"

KILL_ONLY=false
REMOTE_HOST="10.42.0.1"
for arg in "$@"; do
    if [[ "$arg" == "--kill" ]]; then
        KILL_ONLY=true
    else
        REMOTE_HOST="$arg"
    fi
done

SSH="ssh -o ConnectTimeout=5 -o BatchMode=yes ${REMOTE_USER}@${REMOTE_HOST}"

remote_kill() {
    $SSH "pkill -f '${REMOTE_SCRIPT}'" 2>/dev/null
    # pkill returns 1 when no process found -- that's fine
    return 0
}

if $KILL_ONLY; then
    echo "Killing gello_position_server on ${REMOTE_USER}@${REMOTE_HOST} ..."
    remote_kill
    echo "Done."
    exit 0
fi

# Quick connectivity check
echo "Checking connectivity to ${REMOTE_HOST} ..."
if ! ping -c 1 -W 2 "${REMOTE_HOST}" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach ${REMOTE_HOST}. Is the device connected?"
    exit 1
fi
if ! $SSH "echo ok" > /dev/null 2>&1; then
    echo "ERROR: SSH connection failed. Set up key auth with:"
    echo "  ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}"
    exit 1
fi

echo "=== GELLO Server Launcher ==="
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}"
echo ""

# 1. Kill any existing instance
echo "[1/3] Killing existing server (if any) ..."
remote_kill
echo "  Done."

# 2. Copy latest script
echo "[2/3] Copying ${LOCAL_SCRIPT} ..."
scp -q -o ConnectTimeout=5 "${LOCAL_SCRIPT}" "${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_SCRIPT}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy script."
    exit 1
fi
echo "  Copied."

# 3. Start in detached screen
echo "[3/3] Starting server in screen session '${SCREEN_NAME}' ..."
$SSH "screen -dmS ${SCREEN_NAME} python3 ~/${REMOTE_SCRIPT}"
sleep 1

# Verify it's running
if $SSH "pgrep -f '${REMOTE_SCRIPT}'" > /dev/null 2>&1; then
    echo ""
    echo "Server is running."
    echo "  To view logs:  ssh -t ${REMOTE_USER}@${REMOTE_HOST} 'screen -r ${SCREEN_NAME}'"
    echo "  To detach:     Ctrl+A, D"
    echo "  To kill:       bash scripts/start_gello_server.sh --kill"
else
    echo ""
    echo "WARNING: Server does not appear to be running. SSH in to debug:"
    echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'python3 ~/${REMOTE_SCRIPT}'"
fi
