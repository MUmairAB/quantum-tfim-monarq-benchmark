#!/usr/bin/env bash
#SBATCH --job-name=vqe_seeds
#SBATCH --output=logs/vqe_seeds_%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#
# SLURM job array for the VQE seed ensemble: one array index per run (or per
# chunk of runs, see CHUNK below), so independent runs go through the
# scheduler concurrently instead of one after another.
#
# Each run is single-threaded and peaks around 0.6 GB at L=20, so one core and
# 4 GB per task is comfortable. Pick --time from the deepest circuit in the
# block and how busy the partition is; L=20 runs slow down substantially when
# many of them share a node, so leave generous margin.
#
# Submit with the block name and the array size that `--list` reports:
#
#   ./monarq-env/bin/python -m src.vqe_seeds --config configs/vqe_seed_ensemble.yaml \
#       --blocks p1_critical,p1_ferro --list
#   sbatch --array=0-39 --account=<account> scripts/vqe_seeds_array.sh p1_critical,p1_ferro
#
# Accounts: def-mahtabs_cpu on Narval, def-sponsor00 (partition
# cpubase_bycore_b1) on the teaching cluster.
#
# Resumable: a run whose result file already exists is skipped, so a requeued
# or preempted task only redoes what was actually in flight.

set -euo pipefail

BLOCK="${1:?usage: sbatch --array=0-N vqe_seeds_array.sh <block-names> [chunk-size]}"
CHUNK="${2:-1}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# The teaching cluster's shared JupyterLab kernel sets these, and they silently
# redirect imports away from the venv. Harmless no-ops on Narval.
unset PIP_PREFIX EBPYTHONPREFIXES EBPYTHONPREFIXES_PRIORITY || true

# Narval needs the Python module loaded before the venv works; the teaching
# cluster already has one in place.
module load python/3.11.5 2>/dev/null || true

# Pick up whichever venv this cluster has.
#
# Retried rather than tested once: the shared /project filesystem is
# sometimes not visible the instant a task starts, and a single check then
# fails within seconds, silently wasting the array slot.
CANDIDATES="/project/def-mahtabs/umair/quantum-tfim-env $HOME/quantum-tfim-env ./monarq-env"
for attempt in 1 2 3 4 5 6; do
    for candidate in $CANDIDATES; do
        if [ -x "$candidate/bin/python" ]; then
            PYTHON="$candidate/bin/python"
            break 2
        fi
    done
    # Don't sleep after the last attempt — there is nothing left to wait for.
    if [ "$attempt" -lt 6 ]; then
        echo "attempt $attempt: no venv visible yet on $(hostname), waiting ${attempt}0s" >&2
        sleep "${attempt}0"
    fi
done

if [ -z "${PYTHON:-}" ]; then
    # Fail loudly and name what was actually checked, so the next reader does
    # not have to guess which path was missing.
    echo "ERROR: no virtualenv found on $(hostname) after 6 attempts." >&2
    echo "  looked for bin/python under each of:" >&2
    for candidate in $CANDIDATES; do
        echo "    $candidate  (exists: $([ -d "$candidate" ] && echo yes || echo no))" >&2
    done
    echo "  /project mount visible: $([ -d /project/def-mahtabs ] && echo yes || echo NO)" >&2
    exit 1
fi
echo "using $PYTHON on $(hostname), array task ${SLURM_ARRAY_TASK_ID:-none}"

# Tested and made no measurable difference to runtime — set for reproducibility.
export OMP_NUM_THREADS=1

# A JAX CUDA traceback (cuInit(0) failed, error 303) may appear here because
# jax[cuda12] is installed but no GPU is allocated. JAX falls back to CPU and
# lightning.qubit runs in C++ on CPU regardless — it is safe to ignore.
"$PYTHON" -m src.vqe_seeds \
    --config configs/vqe_seed_ensemble.yaml \
    --blocks "$BLOCK" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" \
    --chunk-size "$CHUNK"
