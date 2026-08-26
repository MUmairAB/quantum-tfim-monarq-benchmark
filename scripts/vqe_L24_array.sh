#!/usr/bin/env bash
#SBATCH --job-name=vqe_L24
#SBATCH --output=logs/vqe_L24_%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#
# SLURM job array for the L=24 VQE runs. Same runner and config as
# vqe_seeds_array.sh; only the resource request differs.
#
# One core and 4 GB. Extra OpenMP threads buy very little at this size, and peak
# resident set is 2.0 GB single-threaded. Many independent single-core runs beat
# widening any one of them.
#
# 48 h rather than 24. SLURM picks the partition from --time (b1=3h, b2=12h,
# b3=24h, b4=3d, b5=7d), so a tight --time silently caps the job. Runs on
# identical hardware have ranged 10h38 to 19h45, and there is no mid-run
# checkpoint, so a timeout discards the whole run. 72 h routes to the same
# partition if more margin is ever wanted.
#
# The teaching cluster cannot run these at all — its walltime is capped at 8 h.
#
# Submit with:
#   sbatch --array=0-23 --account=<account> scripts/vqe_L24_array.sh c2_L24
#
# Resumable: a run whose result file already exists is skipped, so a requeued or
# preempted task only redoes what was actually in flight.

set -euo pipefail

BLOCK="${1:?usage: sbatch --array=0-N vqe_L24_array.sh <block-names> [chunk-size]}"
CHUNK="${2:-1}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# The teaching cluster's shared JupyterLab kernel sets these and they silently
# redirect imports away from the venv. Harmless no-ops on Narval.
unset PIP_PREFIX EBPYTHONPREFIXES EBPYTHONPREFIXES_PRIORITY || true

# Narval needs the Python module loaded before the venv works.
module load python/3.11.5 2>/dev/null || true

for candidate in /project/def-mahtabs/umair/quantum-tfim-env "$HOME/quantum-tfim-env" ./monarq-env; do
    if [ -x "$candidate/bin/python" ]; then
        PYTHON="$candidate/bin/python"
        break
    fi
done
: "${PYTHON:?no virtualenv found}"
echo "using $PYTHON on $(hostname), array task ${SLURM_ARRAY_TASK_ID:-none}"

export OMP_NUM_THREADS=1

# A JAX CUDA traceback (cuInit(0) failed, error 303) may appear here because
# jax[cuda12] is installed but no GPU is allocated. JAX falls back to CPU and
# lightning.qubit runs in C++ on CPU regardless — it is safe to ignore.
"$PYTHON" -m src.vqe_seeds \
    --config configs/vqe_seed_ensemble.yaml \
    --blocks "$BLOCK" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" \
    --chunk-size "$CHUNK"
