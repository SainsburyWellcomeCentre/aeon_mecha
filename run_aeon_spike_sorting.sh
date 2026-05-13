#!/bin/bash

# =============================================================================
# AEON Spike Sorting SLURM Script
# =============================================================================
# Usage:  sbatch run_aeon_spike_sorting.sh
# Check job status: squeue --start -j <job_id>
# Cancel job: scancel <job_id>
# =============================================================================

#SBATCH --job-name=aeon-spike-sorting         # job name
#SBATCH --partition=gpu                       # Change to 'cpu' for CPU-only mode
#SBATCH --gres=gpu:a100:1                    # Remove this line for CPU-only mode (options a100, p5000)
#SBATCH --nodes=1                             # node count
#SBATCH --ntasks=1                            # total number of tasks across all nodes
#SBATCH --mem=256G                            # total memory per node
#SBATCH --time=7-08:00:00                     # total run time limit (DD-HH:MM:SS)
#SBATCH --output=slurm_output/%N_%j.out       # output file path
#SBATCH --error=slurm_output/%N_%j.err        # error file path

# Exit on any error
set -e

# Print job information
echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"
echo "================================"

# Create output directory
mkdir -p slurm_output

# Load modules
echo "Loading modules..."
module load uv

# Change to the analysis repo root (where the venv lives).
# Adjust this path if your analysis repo is elsewhere.
cd ~/ProjectAeon/foragingABC_analysis
echo "Working directory: $(pwd)"

# Set PyTorch CUDA memory allocator configuration to free reserved memory
# This helps prevent CUDA out of memory errors during long-running Kilosort4 jobs
# expandable_segments:True allows PyTorch to dynamically expand memory segments to reduce fragmentation
# garbage_collection_threshold:0.6 triggers GC when 60% of reserved memory is unused,
#   which should free the 7.69 GiB reserved but unallocated memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
echo "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6"

# Start resource profiler in the background
PROFILER_PATH="submodules/aeon_mecha/aeon/dj_pipeline/scripts/start_resource_profiler.py"
if [ -f "$PROFILER_PATH" ]; then
    echo "Starting resource profiler..."
    .venv/bin/python "$PROFILER_PATH" -o "./slurm_output/resource_use_${SLURM_JOB_ID}.csv" & PROFILER_PID=$!
    echo "Resource profiler started with PID: $PROFILER_PID"
else
    echo "Resource profiler not found at $PROFILER_PATH, skipping."
    PROFILER_PID=""
fi

# Verify Python script exists
SCRIPT_PATH="submodules/aeon_mecha/aeon/dj_pipeline/scripts/run_aeon_spike_sorting.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script not found: $SCRIPT_PATH"
    exit 1
fi

# Run the spike sorting script
echo "Starting spike sorting..."
.venv/bin/python "$SCRIPT_PATH"

# Stop the profiler
if [ -n "$PROFILER_PID" ]; then
    echo "Stopping resource profiler..."
    kill $PROFILER_PID
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo "=== Job completed successfully ==="
    echo "End Time: $(date)"
    exit 0
else
    echo "=== Job failed ==="
    echo "End Time: $(date)"
    exit 1
fi
