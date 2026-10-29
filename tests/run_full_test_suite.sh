#!/bin/bash
#SBATCH --job-name=aeon-tests
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err

# Full golden-dataset test suite (no GPU needed). Submit from the repo root: sbatch tests/run_full_test_suite.sh
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_output

export TEST_DB_PREFIX="u_${USER}_golden_tests_"
export DJ_REPOSITORY_CONFIG='{"ceph_aeon": "/ceph/aeon"}'

# The streams_schema fixture regenerates aeon/dj_pipeline/streams.py and restores it in teardown.
# That teardown is skipped if pytest is killed, so restore it here on any exit (incl. scancel).
trap 'git checkout -- aeon/dj_pipeline/streams.py' EXIT

module load uv
uv sync --extra spike_sorting --group test-golden || exit 1
# Preflight: log which DB/namespace this run uses and fail fast if it is unsafe or unreachable.
uv run python -c "
import os, datajoint as dj
pref = os.environ['TEST_DB_PREFIX']
assert pref.endswith('_') and 'test' in pref, f'unsafe prefix: {pref}'
dj.conn()
print('host:', dj.config.database.host, '| user:', dj.config.database.user, '| prefix:', pref)
print('existing schemas under prefix:', [s for s in dj.list_schemas() if s.startswith(pref)])
" || exit 1

uv run pytest -v --tb=short -ra
