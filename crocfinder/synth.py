"""Rough recall check with no real sighting: paste a croc cutout onto real tiles at
several sizes and see how often the detector finds it.

  python -m crocfinder.synth VIDEO.OSV --cutout refs/phone_croc.jpg --out out/synth

The paste is crude (soft-edged patch, original water colour), so read the result as
"about how many pixels long a croc must be", not as a precise recall figure.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFilter

from . import ingest
from .detect import Detector
from .geometry import Source, build_grids, default_tiles
from .run import pil, to_tiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cutout", required=True)
    ap.add_argument("--out", default="out/synth")
    ap.add_argument("--sizes", default="24,32,48,64,96,128,192")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--tile-fov", type=float, default=48.0)
    a = ap.parse_args()

    rng = random.Random(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cut = Image.open(a.cutout).convert("RGB")
    tiles = default_tiles(a.tile_fov, 1024, -14.0)
    grids = build_grids(tiles, Source("fisheye_down", radius_frac=0.98), dev)
    det = Detector(a.model, dev, threshold=a.threshold)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dur = ingest.describe(a.video)["duration_s"]
    sizes = [int(s) for s in a.sizes.split(",")]
    hits = {s: [0, 0] for s in sizes}
    best = {s: [] for s in sizes}
    for fi, (t, frame) in enumerate(ingest.frames(a.video, a.frames / dur, streams=(1,))):
        tt = [pil(x) for x in to_tiles(frame, grids)]
        picks = rng.sample(range(len(tt)), 4)
        for s in sizes:
            batch, where = [], []
            for k in picks:
                im = tt[k].copy()
                w = s
                h = max(4, round(cut.height * w / cut.width))
                ang = rng.uniform(0, 360)
                c = cut.resize((w, h), Image.LANCZOS).rotate(ang, expand=True, resample=Image.BICUBIC)
                m = Image.new("L", (w, h), 0)
                ImageDraw.Draw(m).ellipse((0, 0, w - 1, h - 1), fill=255)
                m = m.filter(ImageFilter.GaussianBlur(max(1.0, h / 5))).rotate(ang, expand=True, resample=Image.BICUBIC)
                x = rng.randint(80, im.width - 80 - c.width)
                y = rng.randint(80, im.height - 80 - c.height)
                im.paste(c, (x, y), m)
                batch.append(im)
                where.append((x + c.width / 2, y + c.height / 2))
            for im, (cx, cy), boxes in zip(batch, where, det(batch)):
                found = [b for b in boxes if b.x0 - 8 <= cx <= b.x1 + 8 and b.y0 - 8 <= cy <= b.y1 + 8
                         and max(b.x1 - b.x0, b.y1 - b.y0) < 4 * s + 40]
                hits[s][1] += 1
                if found:
                    hits[s][0] += 1
                    best[s].append(max(b.score for b in found))
                if fi == 0 and (cx, cy) == where[0]:
                    im.crop((int(cx) - 150, int(cy) - 150, int(cx) + 150, int(cy) + 150)).save(out / f"example_{s}px.jpg")
    res = {s: {"found": h, "of": n, "recall": round(h / n, 2),
               "median_score": round(sorted(best[s])[len(best[s]) // 2], 2) if best[s] else None}
           for s, (h, n) in hits.items()}
    (out / "synth.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
