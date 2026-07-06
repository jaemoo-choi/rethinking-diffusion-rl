#!/bin/bash
# Stop the auto-resubmission monitoring

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ ! -f ".monitor_pid" ]; then
    echo "No monitoring process found (no .monitor_pid file)"
    exit 1
fi

monitor_pid=$(cat .monitor_pid)

if ps -p "$monitor_pid" > /dev/null 2>&1; then
    echo "Stopping monitoring process (PID: $monitor_pid)..."
    kill "$monitor_pid"
    
    # Wait a moment and verify it stopped
    sleep 2
    if ps -p "$monitor_pid" > /dev/null 2>&1; then
        echo "Process didn't stop gracefully, forcing..."
        kill -9 "$monitor_pid"
    fi
    
    rm -f .monitor_pid
    echo "✓ Monitoring stopped"
else
    echo "Process $monitor_pid is not running (removing stale PID file)"
    rm -f .monitor_pid
fi

# Clean up marker files
echo "Cleaning up job marker files..."
rm -f .job_marker_*
echo "✓ Done"
