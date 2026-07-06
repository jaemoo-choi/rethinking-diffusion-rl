#!/bin/bash
# gpu_dispatch.sh — Automatic GPU tier dispatch for SLURM training jobs.
#
# Sourceable library. Provides:
#   submit_with_dispatch <script_path>
#       Checks GPU availability and submits the script to the best
#       available tier.  Returns sbatch stdout (for job-ID extraction).
#
# GPU priority: H200x8 > H100x8 > H200x6 > H100x6 > H200x4 > H100x4
#
# Environment variables:
#   ALLOW_H100       — set to "false" to skip H100 tiers (default: true)
#   DISPATCH_FALLBACK — "queue" to submit H200:8 even when nothing is free,
#                       "skip" to return non-zero (default: skip)
#   DISPATCH_PARTITION — SLURM partition to query (default: coe-gpu)
#
# Per-script overrides (set via `export` inside each scripts/*.sh):
#   MIN_GPUS     — minimum GPU count (cascade tiers below this are skipped)
#   ALLOW_H100   — "false" to exclude H100 tiers for this script
#   FORCE_TIER   — pin this script to a specific tier (e.g. "H100:4",
#                  "H200:6"). Bypasses the greedy cascade entirely.
#                  Walltime is derived from GPU count.

DISPATCH_PARTITION="${DISPATCH_PARTITION:-coe-gpu}"
ALLOW_H100="${ALLOW_H100:-true}"
DISPATCH_FALLBACK="${DISPATCH_FALLBACK:-skip}"

_DISPATCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
_DISPATCH_LOG="${_DISPATCH_DIR}/dispatch.log"

# ── helpers ───────────────────────────────────────────────────────────

_dispatch_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$_DISPATCH_LOG"
}

# ── 1. count_free_gpus ────────────────────────────────────────────────
# Usage: eval "$(count_free_gpus "$exclude_list")"
# Sets shell variables: h200_8free  h200_6free  h100_8free  h100_6free

count_free_gpus() {
    local exclude_list="$1"

    python3 - "$DISPATCH_PARTITION" "$exclude_list" <<'PYEOF'
import re, subprocess, sys

partition_filter = sys.argv[1]
exclude_raw = sys.argv[2] if len(sys.argv) > 2 else ""
exclude_set = set()
for part in exclude_raw.split(","):
    node = part.strip()
    if node:
        # strip FQDN suffix if present
        exclude_set.add(node.split(".")[0])

output = subprocess.check_output(["scontrol", "-o", "show", "node"], text=True)

# Counters: nodes with >= N free GPUs of each type
counts = {
    "h200_8free": 0, "h200_6free": 0, "h200_4free": 0,
    "h100_8free": 0, "h100_6free": 0, "h100_4free": 0,
}

for raw_line in output.splitlines():
    line = " ".join(raw_line.split())

    node_m = re.search(r"\bNodeName=([^ ]+)", line)
    part_m = re.search(r"\bPartitions=([^ ]+)", line)
    state_m = re.search(r"\bState=([^ ]+)", line)
    cfg_m = re.search(r"\bCfgTRES=([^ ]+)", line)
    alloc_m = re.search(r"\bAllocTRES=([^ ]+)", line)

    if not (node_m and cfg_m):
        continue

    node = node_m.group(1)
    partitions = part_m.group(1) if part_m else ""
    state = (state_m.group(1) if state_m else "").upper()
    cfg_tres = cfg_m.group(1)
    alloc_tres = alloc_m.group(1) if alloc_m else ""

    # partition filter
    if partition_filter and partition_filter not in partitions.split(","):
        continue

    # skip unhealthy nodes
    if any(s in state for s in ["DOWN", "DRAIN", "NOT_RESPONDING", "FAIL", "POWER"]):
        continue

    # skip excluded nodes
    short_node = node.split(".")[0]
    if short_node in exclude_set:
        continue

    # extract GPU type and count
    gpu_type_m = re.search(r"gres/gpu:([^=,]+)=([0-9]+)", cfg_tres)
    if not gpu_type_m:
        continue
    gpu_type = gpu_type_m.group(1).lower()  # e.g. "h200", "h100"
    total_gpu = int(gpu_type_m.group(2))

    alloc_gpu_m = re.search(rf"gres/gpu:{re.escape(gpu_type)}=([0-9]+)", alloc_tres)
    if not alloc_gpu_m:
        alloc_gpu_m = re.search(r"gres/gpu=([0-9]+)", alloc_tres)
    alloc_gpu = int(alloc_gpu_m.group(1)) if alloc_gpu_m else 0

    free_gpu = total_gpu - alloc_gpu

    if gpu_type == "h200":
        if free_gpu >= 8:
            counts["h200_8free"] += 1
        if free_gpu >= 6:
            counts["h200_6free"] += 1
        if free_gpu >= 4:
            counts["h200_4free"] += 1
    elif gpu_type == "h100":
        if free_gpu >= 8:
            counts["h100_8free"] += 1
        if free_gpu >= 6:
            counts["h100_6free"] += 1
        if free_gpu >= 4:
            counts["h100_4free"] += 1

for k, v in counts.items():
    print(f"{k}={v}")
PYEOF
}

# ── 2a. count_my_pending_per_tier ────────────────────────────────────
# Counts how many of THIS user's currently-pending jobs are queued for each
# tier. The cascade subtracts these from <tier>_free so a sweep doesn't pile
# all jobs into the same tier (which is what was happening: 7 jobs all
# dispatched to H200:4 while H100:4 sat idle, because <tier>_free from
# scontrol does not decrement when our own jobs are queued — only when SLURM
# actually places them).

count_my_pending_per_tier() {
    python3 - "$USER" <<'PYEOF'
import subprocess, re, sys

user = sys.argv[1]
try:
    out = subprocess.check_output(
        ["squeue", "-u", user, "-t", "PENDING", "-h", "-o", "%b"],
        text=True,
    )
except Exception:
    out = ""

counts = {f"my_pending_{t}_{n}": 0 for t in ("h200", "h100") for n in (8, 6, 4)}
for line in out.splitlines():
    m = re.search(r"gpu:(h200|h100):(\d+)", line, re.IGNORECASE)
    if not m:
        continue
    gpu, n = m.group(1).lower(), int(m.group(2))
    if n in (8, 6, 4):
        counts[f"my_pending_{gpu}_{n}"] += 1

for k, v in counts.items():
    print(f"{k}={v}")
PYEOF
}

# ── 2. get_best_gpu_tier ─────────────────────────────────────────────
# Usage: tier=$(get_best_gpu_tier "$exclude_list")
# Returns: "GPU_TYPE:GPU_COUNT:WALL_TIME" or "NONE"

get_best_gpu_tier() {
    local exclude_list="$1"
    local min_gpus="${2:-1}"

    # Query current availability
    local avail
    avail=$(count_free_gpus "$exclude_list")

    # Parse into variables
    local h200_8free=0 h200_6free=0 h200_4free=0 h100_8free=0 h100_6free=0 h100_4free=0
    eval "$avail"

    # Subtract this user's already-pending jobs in each tier so a sweep
    # spreads across tiers instead of piling into the highest-priority one.
    local my_pending
    my_pending=$(count_my_pending_per_tier 2>/dev/null)
    local my_pending_h200_8=0 my_pending_h200_6=0 my_pending_h200_4=0
    local my_pending_h100_8=0 my_pending_h100_6=0 my_pending_h100_4=0
    eval "$my_pending"

    local h200_8eff=$((h200_8free - my_pending_h200_8))
    local h200_6eff=$((h200_6free - my_pending_h200_6))
    local h200_4eff=$((h200_4free - my_pending_h200_4))
    local h100_8eff=$((h100_8free - my_pending_h100_8))
    local h100_6eff=$((h100_6free - my_pending_h100_6))
    local h100_4eff=$((h100_4free - my_pending_h100_4))

    _dispatch_log "Availability: h200_8free=$h200_8free h200_6free=$h200_6free h200_4free=$h200_4free h100_8free=$h100_8free h100_6free=$h100_6free h100_4free=$h100_4free"
    _dispatch_log "MyPending:    h200_8=$my_pending_h200_8 h200_6=$my_pending_h200_6 h200_4=$my_pending_h200_4 h100_8=$my_pending_h100_8 h100_6=$my_pending_h100_6 h100_4=$my_pending_h100_4"
    _dispatch_log "EffectiveFree: h200_8=$h200_8eff h200_6=$h200_6eff h200_4=$h200_4eff h100_8=$h100_8eff h100_6=$h100_6eff h100_4=$h100_4eff"

    # Priority cascade with effective-free counts: interleave H200/H100 at each
    # GPU count. Order: H200:8 > H100:8 > H200:6 > H100:6 > H200:4 > H100:4.
    # Skip tiers below min_gpus or where our own pending queue saturates the
    # apparent free slots. Require ≥2 effective-free per tier: a lone free
    # slot tends to be grabbed by higher-priority jobs before SLURM places ours.
    # ALLOW_GPU_6 (default true) — set to false in the calling script to
    # skip the 6-GPU tiers entirely. Use when num_groups*nipp doesn't divide
    # cleanly by 6 (e.g. num_groups=32, nipp=8 in the new Wan recipe).
    local allow_6="${ALLOW_GPU_6:-true}"

    if [ "$h200_8eff" -gt 1 ] && [ 8 -ge "$min_gpus" ]; then
        echo "H200:8:02:00:00"
    elif [ "$ALLOW_H100" = "true" ] && [ "$h100_8eff" -gt 1 ] && [ 8 -ge "$min_gpus" ]; then
        echo "H100:8:02:00:00"
    elif [ "$allow_6" = "true" ] && [ "$h200_6eff" -gt 1 ] && [ 6 -ge "$min_gpus" ]; then
        echo "H200:6:02:30:00"
    elif [ "$allow_6" = "true" ] && [ "$ALLOW_H100" = "true" ] && [ "$h100_6eff" -gt 1 ] && [ 6 -ge "$min_gpus" ]; then
        echo "H100:6:02:30:00"
    elif [ "$h200_4eff" -gt 1 ] && [ 4 -ge "$min_gpus" ]; then
        echo "H200:4:04:00:00"
    elif [ "$ALLOW_H100" = "true" ] && [ "$h100_4eff" -gt 1 ] && [ 4 -ge "$min_gpus" ]; then
        echo "H100:4:04:00:00"
    # Last-resort fallback: every tier saturated by our own queue. Fall back
    # to absolute free counts (ignore my_pending) so we still queue somewhere.
    elif [ "$h200_8free" -gt 0 ] && [ 8 -ge "$min_gpus" ]; then
        echo "H200:8:02:00:00"
    elif [ "$ALLOW_H100" = "true" ] && [ "$h100_8free" -gt 0 ] && [ 8 -ge "$min_gpus" ]; then
        echo "H100:8:02:00:00"
    elif [ "$allow_6" = "true" ] && [ "$h200_6free" -gt 0 ] && [ 6 -ge "$min_gpus" ]; then
        echo "H200:6:02:30:00"
    elif [ "$allow_6" = "true" ] && [ "$ALLOW_H100" = "true" ] && [ "$h100_6free" -gt 0 ] && [ 6 -ge "$min_gpus" ]; then
        echo "H100:6:02:30:00"
    elif [ "$h200_4free" -gt 0 ] && [ 4 -ge "$min_gpus" ]; then
        echo "H200:4:04:00:00"
    elif [ "$ALLOW_H100" = "true" ] && [ "$h100_4free" -gt 0 ] && [ 4 -ge "$min_gpus" ]; then
        echo "H100:4:04:00:00"
    else
        echo "NONE"
    fi
}

# ── 3. submit_with_dispatch ──────────────────────────────────────────
# Usage: output=$(submit_with_dispatch "/path/to/sweep_p1_X.sh")
# Prints sbatch stdout on success; returns non-zero on failure.

submit_with_dispatch() {
    local script_path="$1"
    local script_name
    script_name=$(basename "$script_path")

    # Extract exclude list and minimum GPU count from the script
    local exclude_list=""
    exclude_list=$(grep -oP '(?<=#SBATCH --exclude=).*' "$script_path" 2>/dev/null | head -1 | tr -d ' ')
    local min_gpus
    min_gpus=$(grep -oP '(?<=export MIN_GPUS=)\d+' "$script_path" 2>/dev/null | head -1)
    min_gpus="${min_gpus:-1}"
    local script_allow_h100
    script_allow_h100=$(grep -oP '(?<=export ALLOW_H100=)\S+' "$script_path" 2>/dev/null | head -1)
    if [ -n "$script_allow_h100" ]; then
        ALLOW_H100="$script_allow_h100"
    fi
    local script_allow_gpu_6
    script_allow_gpu_6=$(grep -oP '(?<=export ALLOW_GPU_6=)\S+' "$script_path" 2>/dev/null | head -1)
    if [ -n "$script_allow_gpu_6" ]; then
        export ALLOW_GPU_6="$script_allow_gpu_6"
    fi

    # ── Per-script tier pinning ─────────────────────────────────────────
    # If the script exports FORCE_TIER=<TYPE>:<COUNT> (e.g. "H100:4"), bypass
    # the greedy cascade and submit to exactly that tier.  Walltime is derived
    # from GPU count (8→2h, 6→2.5h, 4→4h) to stay within the 16 GPU-hr QoS cap.
    local force_tier
    force_tier=$(grep -oP '(?<=export FORCE_TIER=)\S+' "$script_path" 2>/dev/null | head -1)

    local tier
    if [ -n "$force_tier" ]; then
        local ft_type ft_count ft_wall
        IFS=':' read -r ft_type ft_count <<< "$force_tier"
        case "$ft_count" in
            8) ft_wall="02:00:00" ;;
            6) ft_wall="02:30:00" ;;
            4) ft_wall="04:00:00" ;;
            *) ft_wall="02:00:00" ;;
        esac
        tier="${ft_type}:${ft_count}:${ft_wall}"
        _dispatch_log "$script_name -> FORCE_TIER=${force_tier} -> ${tier}"
    else
        tier=$(get_best_gpu_tier "$exclude_list" "$min_gpus")
    fi

    if [ "$tier" = "NONE" ]; then
        if [ "$DISPATCH_FALLBACK" = "queue" ] || [ "$min_gpus" -gt 1 ]; then
            # Smart fallback: prefer the tier with more actual free slots.
            # Old code hardcoded H200:${min_gpus} which created an infinite
            # bounce loop with redispatch_stuck_pending when h100_4 had 1
            # free slot but h200_4 had 0.
            local h100_var="h100_${min_gpus}free"
            local h200_var="h200_${min_gpus}free"
            local h100_avail h200_avail
            h100_avail=$(eval echo "\$${h100_var}")
            h200_avail=$(eval echo "\$${h200_var}")
            local fb_wall
            case "$min_gpus" in
                8) fb_wall="02:00:00" ;;
                6) fb_wall="02:30:00" ;;
                *) fb_wall="04:00:00" ;;
            esac
            if [ "$ALLOW_H100" = "true" ] && [ "${h100_avail:-0}" -gt "${h200_avail:-0}" ]; then
                _dispatch_log "$script_name -> NONE-effective (min_gpus=$min_gpus); fallback H100:${min_gpus} (h100=${h100_avail:-0} > h200=${h200_avail:-0} actually free)"
                tier="H100:${min_gpus}:${fb_wall}"
            else
                _dispatch_log "$script_name -> NONE-effective (min_gpus=$min_gpus); fallback H200:${min_gpus} (h200=${h200_avail:-0} >= h100=${h100_avail:-0})"
                tier="H200:${min_gpus}:${fb_wall}"
            fi
        else
            _dispatch_log "$script_name -> NONE available, skipping"
            echo "No GPUs available, skipping submission" >&2
            return 1
        fi
    fi

    # Parse tier string
    local gpu_type gpu_count wall_time
    IFS=':' read -r gpu_type gpu_count wall_time <<< "$tier"


    # Create temporary modified script
    local tmp_script
    tmp_script=$(mktemp "/tmp/dispatch_${script_name%.sh}_XXXXXX.sh")

    sed \
        -e "s|#SBATCH --gres=gpu:[^:]*:[0-9]*|#SBATCH --gres=gpu:${gpu_type}:${gpu_count}|" \
        -e "s|#SBATCH --time=.*|#SBATCH --time=${wall_time}|" \
        -e "s|export NUM_GPUS=.*|export NUM_GPUS=${gpu_count}|" \
        "$script_path" > "$tmp_script"

    # Submit
    local submit_output
    submit_output=$(sbatch "$tmp_script" 2>&1)
    local rc=$?

    # Clean up temp file
    rm -f "$tmp_script"

    if [ $rc -eq 0 ]; then
        local job_id
        job_id=$(echo "$submit_output" | grep -oP 'Submitted batch job \K\d+')
        _dispatch_log "$script_name -> ${gpu_type}:${gpu_count} (time=${wall_time}) job=${job_id:-?}"
    else
        _dispatch_log "$script_name -> FAILED to submit as ${gpu_type}:${gpu_count}: $submit_output"
    fi

    echo "$submit_output"
    return $rc
}


# ── 4. redispatch_stuck_pending ──────────────────────────────────────
# Scan the user's PENDING (Reason=Resources) jobs whose requested GPU tier
# has 0 free slots, and cancel them when an *alternative* GPU type at the
# same count has free slots. The cancellation removes the corresponding
# .job_marker_*; the next auto_resubmit cycle's check_and_submit will then
# re-submit via submit_with_dispatch, which sees the now-free alt tier.
#
# Cancels at most ONE job per call to avoid stampedes.
# Returns 0 on cancel, 1 if no eligible job.
#
# Usage (from auto_resubmit.sh): redispatch_stuck_pending
redispatch_stuck_pending() {
    eval "$(count_free_gpus "" 2>/dev/null)"

    local jobs
    jobs=$(squeue -u "$USER" -h -o "%i %T %r" 2>/dev/null | awk '$2=="PENDING" && $3=="Resources" {print $1}')
    [[ -z "$jobs" ]] && return 1

    local script_dir="${_DISPATCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    local now_epoch
    now_epoch=$(date +%s)
    # Cooldown: don't redispatch a job that was submitted < this many seconds ago.
    # Prevents tight bounce loops when free counts oscillate cycle-to-cycle.
    local REDISPATCH_COOLDOWN_SEC="${REDISPATCH_COOLDOWN_SEC:-300}"

    local job_id
    for job_id in $jobs; do
        # Get this job's TRES + SubmitTime in one scontrol call
        local tres
        tres=$(scontrol show job "$job_id" -o 2>/dev/null)
        [[ -z "$tres" ]] && continue

        # Cooldown: skip jobs submitted < REDISPATCH_COOLDOWN_SEC ago
        local submit_time submit_epoch
        submit_time=$(echo "$tres" | grep -oP 'SubmitTime=\K\S+')
        if [[ -n "$submit_time" ]]; then
            submit_epoch=$(date -d "$submit_time" +%s 2>/dev/null)
            if [[ -n "$submit_epoch" ]] && (( now_epoch - submit_epoch < REDISPATCH_COOLDOWN_SEC )); then
                continue
            fi
        fi

        local gpu_type gpu_count
        gpu_type=$(echo "$tres" | grep -oP 'gres/gpu:\K[a-z0-9]+(?==)' | head -1)
        gpu_count=$(echo "$tres" | grep -oP 'gres/gpu:[a-z0-9]+=\K[0-9]+' | head -1)
        [[ -z "$gpu_type" || -z "$gpu_count" ]] && continue

        # Free slots in this tier?
        local cur_var="${gpu_type}_${gpu_count}free"
        local cur_free
        cur_free=$(eval echo "\$${cur_var}")
        # If current tier has free slots, SLURM should be placing it — skip.
        [[ "${cur_free:-0}" -gt 0 ]] && continue

        # If the job's source script pins FORCE_TIER, do NOT redispatch:
        # check_and_submit would just re-run submit_with_dispatch, which
        # re-applies FORCE_TIER → resubmits to the same full tier → bounces
        # forever. Skip this job and let SLURM place it when a slot opens.
        local marker script_path script_force_tier=""
        for marker in "${script_dir}/.job_marker_"*; do
            [[ -f "$marker" ]] || continue
            if [[ "$(cat "$marker" 2>/dev/null)" == "$job_id" ]]; then
                script_path=$(find scripts -path scripts/eval -prune -o -name "$(basename "$marker" | sed 's/^.job_marker_//').sh" -print 2>/dev/null | head -1)
                break
            fi
        done
        if [[ -n "${script_path:-}" && -f "${REPO_ROOT:-.}/${script_path}" ]]; then
            script_force_tier=$(grep -oP '(?<=export FORCE_TIER=)\S+' "${REPO_ROOT:-.}/${script_path}" 2>/dev/null | head -1)
        elif [[ -n "${script_path:-}" && -f "${script_path}" ]]; then
            script_force_tier=$(grep -oP '(?<=export FORCE_TIER=)\S+' "${script_path}" 2>/dev/null | head -1)
        fi
        if [[ -n "$script_force_tier" ]]; then
            _dispatch_log "redispatch: job=$job_id has FORCE_TIER=${script_force_tier} — skipping (alt-tier probe would bounce)"
            continue
        fi

        # Probe alt GPU types at the same count for free slots
        local alt_type
        for alt_type in h100 h200; do
            [[ "$alt_type" == "$gpu_type" ]] && continue
            local alt_var="${alt_type}_${gpu_count}free"
            local alt_free
            alt_free=$(eval echo "\$${alt_var}")
            if [[ "${alt_free:-0}" -gt 0 ]]; then
                _dispatch_log "redispatch: job=$job_id stuck on ${gpu_type}:${gpu_count} (0 free); ${alt_type}:${gpu_count}=${alt_free} free — cancelling for fresh dispatch"
                # Find and remove the matching marker file (so next cycle resubmits)
                local marker
                for marker in "${script_dir}/.job_marker_"*; do
                    [[ -f "$marker" ]] || continue
                    if [[ "$(cat "$marker" 2>/dev/null)" == "$job_id" ]]; then
                        rm -f "$marker"
                        _dispatch_log "redispatch: removed marker $(basename "$marker")"
                        break
                    fi
                done
                scancel "$job_id" 2>/dev/null
                # Tiny delay so SLURM updates the queue before the next check_and_submit
                sleep 3
                return 0
            fi
        done
    done
    return 1
}
