#!/usr/bin/env bash
# Run the VQE seed ensemble locally, several runs at a time.
#
# Each run is single-threaded and needs well under 1 GB, so several fit on one
# machine. Throughput is limited by memory bandwidth rather than cores, so it
# rises much more slowly than the worker count; 8 workers is a reasonable
# default on an 8-core machine.
#
# Usage:  scripts/run_seeds_local.sh <block-name> [n-workers] [chunk-size]
#
# Runs under `caffeinate` so the machine does not sleep partway through.

set -euo pipefail
cd "$(dirname "$0")/.."

BLOCK="${1:?usage: run_seeds_local.sh <block-name> [n-workers] [chunk-size]}"
WORKERS="${2:-8}"
CHUNK="${3:-1}"

CONFIG=configs/vqe_seed_ensemble.yaml
PYTHON=./monarq-env/bin/python

# Each run is single-threaded regardless of this, but set it explicitly so
# the worker processes cannot oversubscribe each other.
export OMP_NUM_THREADS=1

N_RUNS=$("$PYTHON" -m src.vqe_seeds --config "$CONFIG" --blocks "$BLOCK" --list | head -1 | awk '{print $1}')
N_TASKS=$(( (N_RUNS + CHUNK - 1) / CHUNK ))

echo "block $BLOCK: $N_RUNS runs -> $N_TASKS tasks of $CHUNK, $WORKERS at a time"

# Feed the indices longest-job-first and let xargs hand them out dynamically:
# a worker that finishes early picks up the next task instead of idling, and
# starting with the expensive runs keeps a 2-hour job off the end of the queue.
# Already-finished tasks are left out of the list entirely.
"$PYTHON" -m src.vqe_seeds --config "$CONFIG" --blocks "$BLOCK" \
    --chunk-size "$CHUNK" --task-order \
  | caffeinate -i xargs -P "$WORKERS" -I{} \
    "$PYTHON" -m src.vqe_seeds --config "$CONFIG" --blocks "$BLOCK" \
    --index {} --chunk-size "$CHUNK"

echo "block $BLOCK finished"
