#!/usr/bin/env bash
# Pull seed-ensemble result files back from Fir.
#
# Same shape as fetch_narval_results.sh and for the same reason: `git push`
# from a cluster does not work, since there is no TTY to prompt for GitHub
# credentials. Results come back over the SSH connection that submitted the
# jobs and get committed from the laptop.
#
# The two clusters differ only in where the repo lives. Fir has no
# /project/def-mahtabs — it uses numeric project directories and ~/projects was
# never created — so the checkout sits in $HOME and the remote path below is
# relative to the remote home directory.
#
# rsync only copies files that are missing, so this is safe to run repeatedly
# while an array is still finishing — each call picks up whatever has landed.
#
# Usage:  scripts/fetch_fir_results.sh [remote-subdir]
#
# Needs the multiplexed SSH connection to be up (see ~/.ssh/cm/). Fir requires
# Duo MFA, so re-establishing it needs someone to approve a push on their
# phone — the socket is shared, don't tear it down.

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_HOST=umair@fir.alliancecan.ca
REMOTE_REPO=quantum-tfim-monarq-benchmark   # relative to $HOME on Fir
SUBDIR="${1:-results/vqe_seeds}"
CONTROL="$HOME/.ssh/cm/%r@%h:%p"

if ! ssh -o ControlPath="$CONTROL" -O check "$REMOTE_HOST" 2>/dev/null; then
    echo "no live SSH connection to Fir — it needs a Duo push approved on the user's phone" >&2
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
