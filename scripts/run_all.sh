#!/usr/bin/env bash
# Scan every .OSV under $CROC_BUCKET_PREFIX (subfolders included) and upload a gallery per video to $CROC_BUCKET_PREFIX/results/.
# Usage: CROC_BUCKET_PREFIX=gs://bucket/path scripts/run_all.sh [extra args for `crocfinder.run scan`]
# Set TAG to keep runs with different settings apart (default: base).
set -euo pipefail
: "${CROC_BUCKET_PREFIX:?set CROC_BUCKET_PREFIX}"
TAG="${TAG:-base}"
WORK="${WORK:-$HOME/crocwork}"
mkdir -p "$WORK/video" "$WORK/refs" "$WORK/out"
gsutil -q -m rsync "$CROC_BUCKET_PREFIX/refs" "$WORK/refs"
# ** matches subfolders too; read line by line so folder names with spaces work
gsutil ls "$CROC_BUCKET_PREFIX/**.OSV" | while IFS= read -r uri; do
  name=$(basename "$uri" .OSV)
  dest="$CROC_BUCKET_PREFIX/results/$TAG/$name"
  if gsutil -q stat "$dest/meta.json"; then echo "skip $name (done)"; continue; fi
  echo "=== $name"
  gsutil -q cp "$uri" "$WORK/video/$name.OSV"
  python3 -m crocfinder.run scan "$WORK/video/$name.OSV" --out "$WORK/out/$TAG/$name" --refs "$WORK/refs" "$@" < /dev/null
  gsutil -q -m rsync -r "$WORK/out/$TAG/$name" "$dest"
  rm -f "$WORK/video/$name.OSV"
done
