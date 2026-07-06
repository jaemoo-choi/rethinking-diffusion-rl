#!/bin/bash
# Start the auto-resubmission monitoring in the background

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if already running.
# Primary guard: any live auto_resubmit.sh process owned by this user. We check
# the actual process table (not just .monitor_pid) so a daemon launched directly
# as `./auto_resubmit.sh`, or a start while .monitor_pid was stale/missing, can't
# slip past and create a second daemon — two daemons race on the markers and
# double-submit jobs. pgrep -f matches the script path in the command line; the
# `nohup ./auto_resubmit.sh` we launch below is the only thing that matches.
running_pids=$(pgrep -u "$USER" -f 'auto_resubmit\.sh' 2>/dev/null)
if [ -n "$running_pids" ]; then
    running_pids=$(echo "$running_pids" | tr '\n' ' ' | sed 's/ *$//')
    echo "Monitoring is already running (PID(s): $running_pids)"
    # Reconcile .monitor_pid with reality so check/stop track a live daemon.
    first_pid=$(echo "$running_pids" | awk '{print $1}')
    echo "$first_pid" > .monitor_pid
    if [ "$(echo "$running_pids" | wc -w)" -gt 1 ]; then
        echo "WARNING: more than one daemon is running — kill the extras:"
        echo "  kill -9 $running_pids   # then restart"
    fi
    echo "To stop it, run: ./stop_monitoring.sh"
    exit 1
fi

# No live daemon found; drop any stale PID file before starting fresh.
rm -f .monitor_pid

echo "Starting auto-resubmission monitoring..."
echo "Logs will be written to: auto_resubmit.log"
echo ""

# Start in background and save PID
nohup ./auto_resubmit.sh > /dev/null 2>&1 &
monitor_pid=$!

echo $monitor_pid > .monitor_pid
echo "✓ Monitoring started (PID: $monitor_pid)"
echo ""
echo "To check status: tail -f auto_resubmit.log"
echo "To stop monitoring: ./stop_monitoring.sh"
echo "To check if running: ./check_monitoring.sh"
