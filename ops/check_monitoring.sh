#!/bin/bash
# Check the status of auto-resubmission monitoring

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== Auto-Resubmission Monitoring Status ==="
echo ""

if [ ! -f ".monitor_pid" ]; then
    echo "Status: NOT RUNNING"
    echo ""
    echo "To start: ./start_monitoring.sh"
    exit 0
fi

monitor_pid=$(cat .monitor_pid)

if ps -p "$monitor_pid" > /dev/null 2>&1; then
    echo "Status: RUNNING (PID: $monitor_pid)"
    echo ""
    
    # Show tracked jobs
    echo "Tracked jobs:"
    for marker in .job_marker_*; do
        if [ -f "$marker" ]; then
            script_id=$(echo "$marker" | sed 's/\.job_marker_//')
            job_id=$(cat "$marker")
            if squeue -j "$job_id" -h &>/dev/null; then
                status=$(squeue -j "$job_id" -h -o "%T")
                echo "  - train${script_id#train}.sh → Job $job_id ($status)"
            else
                echo "  - train${script_id#train}.sh → Job $job_id (completed/failed)"
            fi
        fi
    done
    
    echo ""
    echo "Recent log entries:"
    if [ -f "auto_resubmit.log" ]; then
        tail -n 10 auto_resubmit.log
    else
        echo "  (no log file yet)"
    fi
    
    echo ""
    echo "To view full log: tail -f auto_resubmit.log"
    echo "To stop: ./stop_monitoring.sh"
else
    echo "Status: STOPPED (stale PID file)"
    rm -f .monitor_pid
    echo ""
    echo "To start: ./start_monitoring.sh"
fi
