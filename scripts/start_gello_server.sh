#!/bin/bash
# Launch the GELLO position server on the R1 Lite Teleop onboard computer.
#
# Usage:
#   bash scripts/start_gello_server.sh                              # default host
#   bash scripts/start_gello_server.sh 10.42.0.2                    # custom host
#   bash scripts/start_gello_server.sh --kill                       # just kill, don't restart
#   bash scripts/start_gello_server.sh -- --motor-ids 1 2 3 4 5 6   # forward args to server
#
# What it does:
#   1. Syncs the latest script over (skipped if the remote copy is already
#      identical; done BEFORE killing so a failed copy never leaves the
#      remote with no server running)
#   2. Kills any existing gello_position_server.py on the remote
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
EXTRA_ARGS=()
forward=false
for arg in "$@"; do
    if $forward; then
        EXTRA_ARGS+=("$arg")
    elif [[ "$arg" == "--" ]]; then
        forward=true
    elif [[ "$arg" == "--kill" ]]; then
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

# 1. Sync latest script — BEFORE killing the running server, so a failed
#    copy never leaves the remote with no server running.  The copy is
#    skipped when the remote copy is already byte-identical (the common
#    case), which also avoids tripping host DLP agents that gate scp.
echo "[1/3] Syncing ${LOCAL_SCRIPT} ..."
LOCAL_MD5=$(md5sum "${LOCAL_SCRIPT}" | awk '{print $1}')
REMOTE_MD5=$($SSH "md5sum ~/${REMOTE_SCRIPT} 2>/dev/null" | awk '{print $1}')
if [ -n "${LOCAL_MD5}" ] && [ "${LOCAL_MD5}" = "${REMOTE_MD5}" ]; then
    echo "  Remote copy already up to date (md5 ${LOCAL_MD5:0:8}) — skipping copy."
else
    scp -q -o ConnectTimeout=5 "${LOCAL_SCRIPT}" "${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_SCRIPT}"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to copy script."
        echo "  If the error above says 'Operation not permitted', this machine's"
        echo "  DLP agent (qzhddr) blocked scp from reading the local file."
        echo "  Retrying won't help — ask the machine admin to allowlist this"
        echo "  transfer (internal deploy to the robot at ${REMOTE_HOST})."
        if [ -n "${REMOTE_MD5}" ]; then
            echo "  NOTE: the remote has an OLDER copy (md5 ${REMOTE_MD5:0:8} vs"
            echo "  local ${LOCAL_MD5:0:8}). Not starting it automatically since it"
            echo "  is stale. The running server (if any) was left untouched."
        fi
        exit 1
    fi
    echo "  Copied."
fi

# 2. Kill any existing instance
echo "[2/3] Killing existing server (if any) ..."
remote_kill
echo "  Done."

# 3. Start in detached screen
echo "[3/3] Starting server in screen session '${SCREEN_NAME}' ..."
EXTRA_QUOTED=""
for a in "${EXTRA_ARGS[@]}"; do
    EXTRA_QUOTED+=" $(printf '%q' "$a")"
done
# -L/-Logfile: persist server output on the remote so crashes/wedges are
# diagnosable after the screen session is gone.
$SSH "screen -L -Logfile /home/${REMOTE_USER}/gello_server.log -dmS ${SCREEN_NAME} python3 ~/${REMOTE_SCRIPT}${EXTRA_QUOTED}"
sleep 1

# Verify end-to-end: the old pgrep check self-matched its own ssh command line
# (always "running"), and a live process can still be wedged on a stale client.
# Reading the protocol header proves the server is actually serving.
VERIFY=$(python3 - "$REMOTE_HOST" <<'EOF'
import socket, struct, sys
try:
    s = socket.socket()
    s.settimeout(4)
    s.connect((sys.argv[1], 5555))
    h = b""
    while len(h) < 4:
        c = s.recv(4 - len(h))
        if not c:
            raise RuntimeError("closed")
        h += c
    s.close()
    print(f"ok:{struct.unpack('!I', h)[0]}")
except Exception as e:
    print(f"fail:{e}")
EOF
)
if [[ "$VERIFY" == ok:* ]]; then
    echo ""
    echo "Server is running and streaming (${VERIFY#ok:} joints)."
    echo "  To view logs:  ssh -t ${REMOTE_USER}@${REMOTE_HOST} 'screen -r ${SCREEN_NAME}'"
    echo "  Log file:      ${REMOTE_USER}@${REMOTE_HOST}:~/gello_server.log"
    echo "  To kill:       bash scripts/start_gello_server.sh --kill"
else
    echo ""
    echo "WARNING: Server not serving (probe: ${VERIFY#fail:})."
    if $SSH "pgrep -f '[g]ello_position_server'" > /dev/null 2>&1; then
        echo "  Process IS alive — likely wedged on a stale client. Re-run this script."
    else
        echo "  Process is NOT running. Check the log:"
        echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -30 ~/gello_server.log'"
    fi
fi
