"""Scan one video for crocodile candidates.

  python -m crocfinder.run probe VIDEO
  python -m crocfinder.run tiles VIDEO --t 30 --out out/tiles      # eyeball the reprojection
  python -m crocfinder.run scan  VIDEO --out out/NAME --refs refs/  # sparse pass, then dense pass
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import ingest, report
from .geometry import Source, build_grids, default_tiles, pixel_direction, reproject


def add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="fisheye_down",
                   choices=["equirect", "dual_fisheye", "fisheye_down"])
    p.add_argument("--streams", default="1", help="video stream ordinals, e.g. 0 or 0,1 (one per lens)")
    p.add_argument("--lens-fov", type=float, default=190.0)
    p.add_argument("--swap", action="store_true", help="back lens is on the left")
    p.add_argument("--roll-front", type=float, default=0.0)
    p.add_argument("--roll-back", type=float, default=0.0)
    p.add_argument("--radius-frac", type=float, default=0.98)
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--tile-fov", type=float, default=48.0)
    p.add_argument("--top-pitch", type=float, default=-14.0, help="pitch of the highest ring of tiles")
    p.add_argument("--hwaccel", action="store_true", help="decode on the GPU")


def source_of(a) -> tuple[Source, tuple[int, ...]]:
    return (Source(a.source, a.lens_fov, a.swap, a.roll_front, a.roll_back, a.radius_frac),
            tuple(int(s) for s in a.streams.split(",")))


def to_tiles(frame: np.ndarray, grids: torch.Tensor, stretch: bool = True) -> torch.Tensor:
    f = torch.from_numpy(frame.copy()).to(grids.device).permute(2, 0, 1).float()
    t = reproject(f, grids)                                 # (n,3,S,S)
    if stretch:
        # footage is a flat log profile: stretch contrast using the middle of the frame
        h, w = frame.shape[:2]
        mid = f[:, h // 4:3 * h // 4:16, w // 4:3 * w // 4:16].mean(0).flatten()
        lo, hi = torch.quantile(mid, 0.01), torch.quantile(mid, 0.995)
        t = (t - lo) * (255.0 / max(float(hi - lo), 32.0))
    return t.clamp(0, 255).byte()


def pil(t: torch.Tensor) -> Image.Image:
    return Image.fromarray(t.permute(1, 2, 0).cpu().numpy())


def cmd_probe(a) -> None:
    print(json.dumps(ingest.describe(a.video), indent=2))


def cmd_tiles(a) -> None:
    src, streams = source_of(a)
    tiles = default_tiles(a.tile_fov, a.tile_size, a.top_pitch)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grids = build_grids(tiles, src, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, frame = next(ingest.frames(a.video, 1, start=a.t, end=a.t + 2, streams=streams, hwaccel=a.hwaccel))
    Image.fromarray(frame).resize((frame.shape[1] // 4, frame.shape[0] // 4)).save(out / "frame_quarter.jpg")
    for tile, img in zip(tiles, to_tiles(frame, grids)):
        pil(img).save(out / f"yaw{tile.yaw:+04.0f}_pitch{tile.pitch:+03.0f}.jpg", quality=90)
    print(f"{len(tiles)} tiles from t={t:.1f}s written to {out}")


def scan_pass(a, det, tiles, grids, src_streams, fps, windows, crops_dir, tag) -> tuple[list[dict], dict]:
    """Run the detector over (start, end) windows. Returns candidates and timing stats."""
    cands: list[dict] = []
    n_frames = 0
    t0 = time.time()
    for start, end in windows:
        for t, frame in ingest.frames(a.video, fps, start=start, end=end, streams=src_streams,
                                      hwaccel=a.hwaccel):
            n_frames += 1
            tt = to_tiles(frame, grids)
            imgs: dict[int, Image.Image] = {}
            for i in range(0, len(tt), a.batch):
                for k, boxes in enumerate(det.tiles(tt[i:i + a.batch]), start=i):
                    if boxes:
                        imgs[k] = pil(tt[k])
                    for j, b in enumerate(boxes):
                        if (b.x1 - b.x0) * (b.y1 - b.y0) > a.max_box_frac * imgs[k].width * imgs[k].height:
                            continue      # a box covering most of the tile is the lake, not an animal
                        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
                        yaw, pitch = pixel_direction(tiles[k], cx, cy)
                        name = f"{tag}_{t:09.2f}_{k:02d}_{j}"
                        pad = 0.75 * max(b.x1 - b.x0, b.y1 - b.y0, 96)
                        box = (max(0, cx - pad), max(0, cy - pad),
                               min(imgs[k].width, cx + pad), min(imgs[k].height, cy + pad))
                        crop = imgs[k].crop(box)
                        crop.save(crops_dir / f"{name}.jpg", quality=92)
                        ctx = imgs[k].copy()
                        ImageDraw.Draw(ctx).rectangle((b.x0, b.y0, b.x1, b.y1), outline=(255, 40, 40), width=3)
                        ctx.save(crops_dir / f"{name}_ctx.jpg", quality=80)
                        cands.append({"t": t, "yaw": yaw, "pitch": pitch, "label": b.label,
                                      "det_score": b.score, "tile": k,
                                      "box_px": f"{b.x1 - b.x0:.0f}x{b.y1 - b.y0:.0f}",
                                      "crop": f"crops/{name}.jpg", "context": f"crops/{name}_ctx.jpg",
                                      "_crop": crop})
            if n_frames % 50 == 0:
                el = time.time() - t0
                print(f"[{tag}] t={t:8.1f}s frames={n_frames} cands={len(cands)} "
                      f"{n_frames / el:.2f} frames/s", flush=True)
    el = time.time() - t0
    return cands, {"frames": n_frames, "seconds": round(el, 1)}


def merge_windows(times: list[float], half: float, duration: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for t in sorted(times):
        s, e = max(0.0, t - half), min(duration, t + half)
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def cmd_scan(a) -> None:
    from .detect import Detector
    from .verify import Verifier, mark_persistent

    src, streams = source_of(a)
    info = ingest.describe(a.video)
    duration = info["duration_s"]
    end = min(duration, a.end) if a.end else duration
    tiles = default_tiles(a.tile_fov, a.tile_size, a.top_pitch)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grids = build_grids(tiles, src, dev)
    out = Path(a.out)
    crops = out / "crops"
    crops.mkdir(parents=True, exist_ok=True)

    det = Detector(a.model, dev, threshold=a.threshold)
    refs = [Image.open(p).convert("RGB") for p in sorted(Path(a.refs).glob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}] if a.refs else []
    ver = Verifier(refs, device=dev)

    def verify(cands: list[dict]) -> None:
        for i in range(0, len(cands), 64):
            chunk = cands[i:i + 64]
            for c, (ts, rs) in zip(chunk, ver.score([c.pop("_crop") for c in chunk])):
                c["text_score"], c["ref_score"] = ts, rs

    print(f"{a.video}: {duration:.0f}s, {len(tiles)} tiles/frame, sparse pass at {a.fps} fps")
    sparse, st1 = scan_pass(a, det, tiles, grids, streams, a.fps, [(a.start, end)], crops, "s")
    verify(sparse)
    mark_persistent(sparse, max_dt=2.5 / a.fps)

    # dense pass: every frame around candidates the verifier did not reject
    keep = [c for c in sparse if c["text_score"] >= a.dense_text or c["det_score"] >= a.dense_det]
    windows = merge_windows([c["t"] for c in keep], a.dense_half, duration)
    dense, st2 = [], {"frames": 0, "seconds": 0}
    if windows and not a.no_dense:
        print(f"dense pass: {len(windows)} windows, {sum(e - s for s, e in windows):.0f}s of video")
        dense, st2 = scan_pass(a, det, tiles, grids, streams, None, windows, crops, "d")
        verify(dense)
        native = info["video_streams"][0]["fps"] or 30
        mark_persistent(dense, max_dt=2.5 / native)

    meta = {"duration_s": round(duration), "sparse_fps": a.fps, "model": det.name,
            "sparse_frames": st1["frames"], "sparse_seconds": st1["seconds"],
            "dense_frames": st2["frames"], "dense_seconds": st2["seconds"],
            "tiles_per_frame": len(tiles), "reference_photos": len(refs)}
    allc = sparse + dense
    # the gallery is for a human: cap it, best first
    for c in allc:
        c["rank_score"] = report.rank_score(c)
    allc.sort(key=lambda c: -c["rank_score"])
    for c in allc[a.max_report:]:
        for k in ("crop", "context"):
            (out / c[k]).unlink(missing_ok=True)
    report.write(out, Path(a.video).name, allc[:a.max_report], meta)
    (out / "meta.json").write_text(json.dumps({**meta, "total_candidates": len(allc)}, indent=2))
    print(json.dumps({**meta, "total_candidates": len(allc)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("video"); p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("tiles"); p.add_argument("video"); p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--out", default="out/tiles"); add_source_args(p); p.set_defaults(fn=cmd_tiles)
    p = sub.add_parser("scan"); p.add_argument("video"); p.add_argument("--out", required=True)
    p.add_argument("--refs", help="folder of reference croc photos (the phone photo)")
    p.add_argument("--model", default="owlv2")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.1, help="detector score threshold (low = high recall)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--dense-half", type=float, default=2.0, help="seconds either side of a hit to scan densely")
    p.add_argument("--dense-text", type=float, default=0.5)
    p.add_argument("--dense-det", type=float, default=0.3)
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--max-box-frac", type=float, default=0.35)
    p.add_argument("--max-report", type=int, default=500)
    add_source_args(p); p.set_defaults(fn=cmd_scan)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
