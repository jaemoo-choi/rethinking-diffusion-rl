#!/bin/bash
# Auto-resubmission script for training jobs
# Checks every few seconds if jobs are submitted and resubmits if needed

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"
# cd to repo root so relative paths like scripts/... resolve correctly
cd "$REPO_ROOT"

# Source GPU dispatch library for automatic tier selection
source "$SCRIPT_DIR/gpu_dispatch.sh"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log file lives in ops/ alongside this script
LOGFILE="$SCRIPT_DIR/auto_resubmit.log"
CHECK_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-10}"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

check_and_submit() {
    local script_name=$1
    local unique_id=$2  # Used to track this specific script
    
    # Create a marker file to track if this script has an active job
    local marker_file="$SCRIPT_DIR/.job_marker_${unique_id}"
    local done_file="$SCRIPT_DIR/.job_done_${unique_id}"
    local job_id=""

    # If this run previously FINISHED (train.py ran through its full epoch loop,
    # i.e. SLURM state COMPLETED), never resubmit it again. To re-enable the run
    # (e.g. after raising MAX_EPOCHS), delete ops/.job_done_<unique_id>.
    if [ -f "$done_file" ]; then
        echo -e "${GREEN}✓ $script_name previously finished — skipping (rm $(basename "$done_file") to re-enable)${NC}"
        return 0
    fi

    # First, check if marker file exists and get job ID
    if [ -f "$marker_file" ]; then
        job_id=$(cat "$marker_file")
        # Verify if this job ID is still visible in the queue.
        # Note: `squeue` often returns exit code 0 even for missing job IDs,
        # so we must check for non-empty output instead of command status.
        job_status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null | head -n 1 | tr -d '[:space:]')
        if [ -n "$job_status" ]; then
            echo -e "${GREEN}✓ $script_name (Job ID: $job_id) is $job_status${NC}"
            log_message "$script_name (Job ID: $job_id) is $job_status"
            return 0
        else
            # Job left the queue. Decide WHY before resubmitting: only resubmit
            # if it did NOT finish. train.py exits 0 when it completes its full
            # epoch loop -> SLURM COMPLETED -> finished, do not resubmit. A
            # wall-time kill is TIMEOUT, a crash is FAILED, etc. -> not finished,
            # resubmit. Query the final accounting state via sacct (-X = the job
            # allocation, not .batch/.extern steps).
            local final_state
            final_state=$(sacct -j "$job_id" --format=State%30 -X -n 2>/dev/null | head -n 1 | tr -d '[:space:]')
            case "$final_state" in
                ""|RUNNING|COMPLETING|PENDING|REQUEUED|RESIZING|SUSPENDED)
                    # Accounting hasn't settled yet — re-check next cycle, keep marker.
                    echo -e "${YELLOW}… $script_name (Job $job_id) left queue, sacct='$final_state' (transient) — waiting${NC}"
                    log_message "$script_name (Job $job_id) left queue, sacct state='$final_state' transient; waiting"
                    return 0
                    ;;
                COMPLETED)
                    # Training finished (reached MAX_EPOCHS). Stop resubmitting.
                    rm -f "$marker_file"
                    touch "$done_file"
                    echo -e "${GREEN}✓ $script_name (Job $job_id) COMPLETED — training finished, not resubmitting${NC}"
                    log_message "$script_name (Job $job_id) COMPLETED — finished; wrote $(basename "$done_file"), not resubmitting"
                    return 0
                    ;;
                *)
                    # TIMEOUT / FAILED / CANCELLED / NODE_FAIL / OUT_OF_MEMORY / etc.
                    rm -f "$marker_file"
                    echo -e "${YELLOW}⚠ $script_name (Job $job_id) ended as '$final_state' — not finished, resubmitting${NC}"
                    log_message "$script_name (Job $job_id) ended as '$final_state' — not finished, will resubmit"
                    ;;
            esac
        fi
    fi
    
    # Check if there are ANY jobs running for this user that might be from this script
    # Look for jobs with the same job name and check their working directory
    job_name=$(grep "^#SBATCH --job-name=" "$script_name" 2>/dev/null | head -1 | sed 's/.*=//' | tr -d ' ')
    if [ -n "$job_name" ]; then
        # Get all jobs with this job name for the current user
        existing_jobs=$(squeue -u $USER -n "$job_name" -h -o "%i %T" 2>/dev/null)
        
        if [ -n "$existing_jobs" ]; then
            # Found existing jobs with same name - assume one belongs to this script
            # (Conservative approach: don't submit if ANY job with this name exists)
            first_job=$(echo "$existing_jobs" | head -1)
            job_id=$(echo "$first_job" | awk '{print $1}')
            job_status=$(echo "$first_job" | awk '{print $2}')
            
            # Save this job ID to marker for future checks
            echo "$job_id" > "$marker_file"
            
            echo -e "${GREEN}✓ $script_name - Found existing job (ID: $job_id, Status: $job_status)${NC}"
            log_message "$script_name - Found existing job (ID: $job_id, Status: $job_status) - not resubmitting"
            return 0
        fi
    fi
    
    # Check if script exists before submitting
    if [ ! -f "$script_name" ]; then
        # Script doesn't exist yet - skip silently
        return 0
    fi

    # No job found in queue
    echo -e "${YELLOW}⚠ $script_name has no active job${NC}"
    log_message "$script_name has no active job - submitting..."

    # Availability snapshot BEFORE submit
    local avail_before
    avail_before=$(count_free_gpus "" 2>/dev/null | tr '\n' ' ')
    log_message "   availability before: $avail_before"

    # Submit the job via GPU dispatch (auto-selects best available GPU tier)
    submit_output=$(submit_with_dispatch "$script_name" 2>&1)
    if [ $? -eq 0 ]; then
        # Extract job ID from sbatch output (format: "Submitted batch job 123456")
        new_job_id=$(echo "$submit_output" | grep -oP 'Submitted batch job \K\d+')
        if [ -n "$new_job_id" ]; then
            echo "$new_job_id" > "$marker_file"
            echo -e "${GREEN}✓ Submitted $script_name (Job ID: $new_job_id)${NC}"
            log_message "Successfully submitted $script_name: $submit_output"

            # Tier that the dispatcher picked (pulled from dispatch.log for this job)
            local chosen_tier
            chosen_tier=$(grep "job=${new_job_id}$" "$_DISPATCH_LOG" 2>/dev/null | tail -n 1 | grep -oP '-> \K[^ ]+')
            [ -n "$chosen_tier" ] && log_message "   dispatched tier: $chosen_tier"

            # Wait briefly for SLURM to register the submission, then report state
            sleep 10
            local new_state
            new_state=$(squeue -j "$new_job_id" -h -o "%T %R" 2>/dev/null | head -n 1)
            if [ -z "$new_state" ]; then
                log_message "   state after submit: NOT IN QUEUE (job may have finished/failed)"
            else
                log_message "   state after submit: ${new_state}"
                echo -e "${GREEN}  → ${new_state}${NC}"
            fi

            # Availability snapshot AFTER submit (shows consumption)
            local avail_after
            avail_after=$(count_free_gpus "" 2>/dev/null | tr '\n' ' ')
            log_message "   availability after:  $avail_after"
        else
            echo -e "${GREEN}✓ Submitted $script_name${NC}"
            log_message "Successfully submitted $script_name: $submit_output"
        fi
    else
        echo -e "${RED}✗ Failed to submit $script_name${NC}"
        log_message "ERROR: Failed to submit $script_name: $submit_output"
    fi
}

# Main monitoring loop
log_message "=== Auto-resubmit script started ==="

while true; do
    echo ""
    echo "================================================"
    echo "Checking jobs at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================"

    # If any job is stuck PENDING with Reason=Resources on a tier that has
    # 0 free slots, but an alt GPU type at the same count is free, cancel
    # that job (and its marker) so check_and_submit below re-dispatches it
    # via submit_with_dispatch onto the now-free tier. One per cycle.
    if redispatch_stuck_pending; then
        log_message "  redispatched a stuck PENDING job (alt tier became free)"
    fi

    # Check each training script
    # Using unique IDs to track each script independently

    # SD3.5-medium GenEval loss-method ablation (EPG / PEPG / PAR and their
    # no-ratio variants), 360 epochs, reference elbo_ode PEPG backdrop
    # (ADV=exact, BETA=1e-4 Girsanov KL, ALPHA=1e-3, DECAY_TYPE=1, LR=3e-4).
    check_and_submit "scripts/sd3/sd3_geneval_epg.sh" "sd3_geneval_epg"
    sleep 15
    check_and_submit "scripts/sd3/sd3_geneval_pepg.sh" "sd3_geneval_pepg"
    sleep 15
    check_and_submit "scripts/sd3/sd3_geneval_par.sh" "sd3_geneval_par"
    sleep 15
    check_and_submit "scripts/sd3/sd3_geneval_epg_noratio.sh" "sd3_geneval_epg_noratio"
    sleep 15
    check_and_submit "scripts/sd3/sd3_geneval_pepg_noratio.sh" "sd3_geneval_pepg_noratio"
    sleep 15

    # FLUX GenEval (original Flow-GRPO objective, stochastic flow SDE sampler).
    check_and_submit "scripts/flux/flux_flow_grpo.sh" "flux_flow_grpo"
    sleep 15

    echo "Next check in ${CHECK_INTERVAL_SECONDS} seconds..."
    echo "================================================"
    log_message "Check cycle completed. Sleeping for ${CHECK_INTERVAL_SECONDS} seconds."

    sleep "$CHECK_INTERVAL_SECONDS"
done
