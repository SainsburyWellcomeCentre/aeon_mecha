#!/bin/bash

#SBATCH --job-name=aeon-spike-sorting         # job name
#SBATCH --partition=gpu                       # partition (queue) gpu_branco
#SBATCH --gres=gpu:a100:1                     # request specific gpu type
#SBATCH --nodes=1                             # node count
#SBATCH --ntasks=1                            # total number of tasks across all nodes
#SBATCH --mem=128G                            # total memory per node
#SBATCH --time=3-00:00:00                     # total run time limit (DD-HH:MM:SS)
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

# Load modules and activate environment (update according to your conda environment name!)
echo "Loading modules..."
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Activating conda environment..."
if ! conda activate aeon_env; then
    echo "ERROR: Failed to activate conda environment 'aeon_env'"
    exit 1
fi

# Verify Python script exists
SCRIPT_PATH="./aeon/dj_pipeline/scripts/run_aeon_spike_sorting.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script not found: $SCRIPT_PATH"
    exit 1
fi

# Run the spike sorting script
echo "Starting spike sorting..."
python "$SCRIPT_PATH"

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
