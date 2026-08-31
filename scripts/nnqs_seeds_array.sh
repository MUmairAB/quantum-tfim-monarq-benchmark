#!/usr/bin/env bash
#SBATCH --job-name=nnqs_seeds
#SBATCH --output=logs/nnqs_seeds_%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#
# SLURM job array for the NNQS seed ensemble, alongside scripts/vqe_seeds_array.sh
# and structured the same way.
#
# These runs are short — about 14 s each at the published sampling budget, up to
# roughly 8 minutes for the 16384-sample runs at L=24 — so one run per array
# index would spend more time in the scheduler than in NetKet. Always pass a
# chunk size, and pick it so a task lands well inside 3 h:
#
#   e1_seed_ensemble      1344 runs, chunk 32 -> 42 tasks of about 7 min
#   e2_nsamples_ablation    96 runs, chunk 4  -> 24 tasks of up to about 32 min
#
# Submit with the block name and the array size that `--list` reports:
#
#   ./monarq-env/bin/python -m src.nnqs_seeds --config configs/nnqs_seed_ensemble.yaml \
#       --blocks e1_seed_ensemble --chunk-size 32 --list
#   sbatch --array=0-41 --account=<account> scripts/nnqs_seeds_array.sh e1_seed_ensemble 32
#
# Account: def-mahtabs_cpu on both Narval and Fir.
#
# Resumable: a run whose result file already exists is skipped, so a requeued
# or preempted task only redoes what was actually in flight.

set -euo pipefail

BLOCK="${1:?usage: sbatch --array=0-N nnqs_seeds_array.sh <block-names> [chunk-size]}"
CHUNK="${2:-1}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# The teaching cluster's shared JupyterLab kernel sets these, and they silently
# redirect imports away from the venv. Harmless no-ops on Narval and Fir.
unset PIP_PREFIX EBPYTHONPREFIXES EBPYTHONPREFIXES_PRIORITY || true

module load python/3.11.5 2>/dev/null || true

# Pick up whichever venv this cluster has. Retried rather than tested once: the
# shared /project filesystem is sometimes not visible the instant a task starts.
CANDIDATES="/project/def-mahtabs/umair/quantum-tfim-env $HOME/quantum-tfim-env ./monarq-env"
for attempt in 1 2 3 4 5 6; do
    for candidate in $CANDIDATES; do
        if [ -x "$candidate/bin/python" ]; then
            PYTHON="$candidate/bin/python"
            break 2
        fi
    done
    if [ "$attempt" -lt 6 ]; then
        echo "attempt $attempt: no venv visible yet on $(hostname), waiting ${attempt}0s" >&2
        sleep "${attempt}0"
    fi
done

if [ -z "${PYTHON:-}" ]; then
    echo "ERROR: no virtualenv found on $(hostname) after 6 attempts." >&2
    echo "  looked for bin/python under each of:" >&2
    for candidate in $CANDIDATES; do
        echo "    $candidate  (exists: $([ -d "$candidate" ] && echo yes || echo no))" >&2
    done
    echo "  /project mount visible: $([ -d /project/def-mahtabs ] && echo yes || echo NO)" >&2
    exit 1
fi
echo "using $PYTHON on $(hostname), array task ${SLURM_ARRAY_TASK_ID:-none}"

# NNQS is the one track that actually threads. NetKet runs on JAX, and both JAX
# and the BLAS underneath it size their thread pools from the machine's core
# count rather than from the cgroup, so on a 1-core allocation they oversubscribe
# and the run gets slower, not faster. Pin all of them to one thread.
#
# This is why the VQE scripts get away with OMP_NUM_THREADS alone: lightning's
# adjoint does not parallelise at these sizes, so there was nothing to pin.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# A JAX CUDA traceback (cuInit(0) failed, error 303) may appear here because
# jax[cuda12] is installed but no GPU is allocated. JAX falls back to CPU and
# NetKet runs on CPU regardless — it is safe to ignore.
"$PYTHON" -m src.nnqs_seeds \
    --config configs/nnqs_seed_ensemble.yaml \
    --blocks "$BLOCK" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" \
    --chunk-size "$CHUNK"
