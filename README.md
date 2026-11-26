# dji-estate-croc

Searches DJI Avata 360 footage for crocodiles.

The Avata 360 records `.OSV` files holding two 3840x3840 fisheye video streams: one lens
looks up, one looks down. Only the downward lens (video stream 1) sees the ground and water,
so that is the one scanned.

## How it works

1. **Decode** the downward lens at about 5 frames per second (`crocfinder/ingest.py`).
2. **Reproject** each fisheye frame into flat, overlapping perspective tiles at native pixel
   density, so small animals are not shrunk (`crocfinder/geometry.py`).
3. **Detect** with a zero-shot, text-prompted detector (OWLv2 or Grounding DINO) at a low
   threshold. No training data is needed (`crocfinder/detect.py`).
4. **Verify** each candidate crop with an image-text model, compare it to reference croc
   photos, and check that it shows up in neighbouring frames (`crocfinder/verify.py`).
5. **Dense pass**: every frame is scanned for a few seconds around surviving candidates.
6. **Report**: a ranked HTML gallery and a CSV with timecodes (`crocfinder/report.py`).

A person makes the final call by looking at the top of the gallery.

## Usage

```bash
pip install -r requirements.txt        # also needs ffmpeg on PATH
python -m crocfinder.run probe VIDEO.OSV
python -m crocfinder.run tiles VIDEO.OSV --t 20 --out out/tiles     # check the reprojection by eye
python -m crocfinder.run scan  VIDEO.OSV --out out/VIDEO --refs refs/
pytest
```

Videos, frames and results are never committed. Bucket and project names live in an
untracked `.env` (see `.env.example`).

## Limits

- A fully submerged animal, or one under tree cover, cannot be seen by any model.
- The detector has not been trained on this footage; treat a clean gallery as
  "nothing found by this method", not proof that no crocodile is present.
