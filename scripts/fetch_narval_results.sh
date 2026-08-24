#!/usr/bin/env bash
# Pull seed-ensemble result files back from Narval.
#
# `git push` from the cluster does not work — there is no TTY to prompt for
# credentials, so it fails with "could not read Username for github.com".
# Rather than putting a token on the cluster, results come back over the same
# SSH connection that submitted the jobs, and get committed from the laptop.
#
# rsync only copies files that are missing or newer, so this is safe to run
# repeatedly while an array is still finishing — each call picks up whatever
# has landed since the last one.
#
# Usage:  scripts/fetch_narval_results.sh [remote-subdir]
#
# Needs the multiplexed SSH connection to be up (see ~/.ssh/cm/). Narval
# requires Duo MFA, so re-establishing it needs someone to approve a push on
# their phone — the socket is shared, don't tear it down.

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_HOST=umair@narval.alliancecan.ca
REMOTE_REPO=/project/def-mahtabs/umair/quantum-tfim-monarq-benchmark
SUBDIR="${1:-results/vqe_seeds}"
CONTROL="$HOME/.ssh/cm/%r@%h:%p"

if ! ssh -o ControlPath="$CONTROL" -O check "$REMOTE_HOST" 2>/dev/null; then
    echo "no live SSH connection to Narval — it needs a Duo push approved on the user's phone" >&2
    exit 1
fi

before=$(find "$SUBDIR" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')

# --ignore-existing: a result file is immutable once written, so never let a
# re-copy overwrite one we already have locally.
rsync -av --ignore-existing \
    -e "ssh -o ControlPath=$CONTROL" \
    "$REMOTE_HOST:$REMOTE_REPO/$SUBDIR/" "$SUBDIR/"

after=$(find "$SUBDIR" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "$SUBDIR: $before -> $after files ($((after - before)) new)"
