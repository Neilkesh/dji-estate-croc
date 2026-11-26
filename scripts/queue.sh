#!/usr/bin/env bash
# Unattended run on the GPU VM: quick pass over every video, a recall check, a 2x zoomed pass,
# then power the VM off so it stops costing money. Everything is uploaded to the bucket as it finishes.
# Usage: CROC_BUCKET_PREFIX=gs://bucket/path nohup scripts/queue.sh > ~/queue.log 2>&1 &
set -uo pipefail
: "${CROC_BUCKET_PREFIX:?set CROC_BUCKET_PREFIX}"
cd "$(dirname "$0")/.."
WORK="${WORK:-$HOME/crocwork}"

TAG=base scripts/run_all.sh --no-dense --batch 10

# recall check on the first video in the bucket
mkdir -p "$WORK/video" "$WORK/refs"
first=$(gsutil ls "$CROC_BUCKET_PREFIX/**.OSV" | head -1)
gsutil -q cp "$first" "$WORK/video/synth.OSV"
for fov in 48 24; do
  python3 -m crocfinder.synth "$WORK/video/synth.OSV" --cutout "$WORK/refs/phone_croc.jpg" \
    --out "$WORK/out/synth_fov$fov" --tile-fov $fov < /dev/null
  gsutil -q -m rsync -r "$WORK/out/synth_fov$fov" "$CROC_BUCKET_PREFIX/results/synth_fov$fov"
done
rm -f "$WORK/video/synth.OSV"

TAG=zoom2x scripts/run_all.sh --no-dense --batch 10 --tile-fov 24 --fps 2
# videos uploaded while the above was running
TAG=base scripts/run_all.sh --no-dense --batch 10
TAG=zoom2x scripts/run_all.sh --no-dense --batch 10 --tile-fov 24 --fps 2

gsutil -q cp ~/queue.log "$CROC_BUCKET_PREFIX/results/queue.log"
[ "${NO_SHUTDOWN:-0}" = 1 ] || sudo shutdown -h now
