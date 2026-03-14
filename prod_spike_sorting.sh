#!/bin/bash

# =============================================================================
# Production Spike Sorting SLURM Script
# =============================================================================
# Usage:  sbatch prod_spike_sorting.sh
# Status: squeue --start -j <job_id>
# Cancel: scancel <job_id>
#
# Submit ONE block per SLURM job for 30h blocks.
# Edit prod_spike_sorting.py keys before each submission.
# =============================================================================

#SBATCH --job-name=aeon-prod-spike-sort
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --time=7-08:00:00
#SBATCH --output=slurm_output/%N_%j.out
#SBATCH --error=slurm_output/%N_%j.err

set -e

echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"
echo "================================"

mkdir -p slurm_output

echo "Loading modules..."
module load uv

# PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
echo "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6"

# Start resource profiler
echo "Starting resource profiler..."
uv run python ./aeon/dj_pipeline/scripts/start_resource_profiler.py -o "./slurm_output/resource_use_${SLURM_JOB_ID}.csv" & PROFILER_PID=$!
echo "Resource profiler PID: $PROFILER_PID"

# Run spike sorting
SCRIPT_PATH="./aeon/dj_pipeline/scripts/prod_spike_sorting.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "Starting spike sorting..."
uv run python "$SCRIPT_PATH"

# Cleanup
echo "Stopping resource profiler..."
kill $PROFILER_PID 2>/dev/null || true

echo "=== Job completed ==="
echo "End Time: $(date)"
