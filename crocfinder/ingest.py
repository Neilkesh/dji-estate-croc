"""Probe and decode videos with ffmpeg. Frames come out as (t_seconds, HxWx3 uint8)."""
from __future__ import annotations

import json
import subprocess
from typing import Iterator

import numpy as np


def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def video_streams(info: dict) -> list[dict]:
    return [s for s in info["streams"] if s.get("codec_type") == "video"
            and s.get("disposition", {}).get("attached_pic", 0) == 0]


def _fps(stream: dict) -> float:
    num, den = stream.get("avg_frame_rate", "0/1").split("/")
    return float(num) / float(den) if float(den) else 0.0


def describe(path: str) -> dict:
    """Short summary used to decide the source layout."""
    info = probe(path)
    vs = video_streams(info)
    return {
        "duration_s": float(info["format"].get("duration", 0)),
        "size_gb": int(info["format"].get("size", 0)) / 1e9,
        "video_streams": [
            {"index": s["index"], "codec": s["codec_name"], "width": s["width"],
             "height": s["height"], "fps": round(_fps(s), 3), "pix_fmt": s.get("pix_fmt")}
            for s in vs],
        "other_streams": [f'{s.get("codec_type")}:{s.get("codec_name")}' for s in info["streams"]
                          if s not in vs],
    }


def frames(path: str, fps: float | None, start: float = 0.0, end: float | None = None,
           streams: tuple[int, ...] = (0,), hwaccel: bool = False,
           scale: tuple[int, int] | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield frames at `fps` (None = every frame) between start and end seconds.

    `streams` are video-stream ordinals. Two ordinals (one per lens) are placed
    side by side, so downstream code always sees one frame.
    """
    info = probe(path)
    vs = video_streams(info)
    sel = [vs[i] for i in streams]
    w, h = (scale if scale else (sel[0]["width"], sel[0]["height"]))
    native = _fps(sel[0])
    out_fps = fps if fps else native

    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", path]
    chain = []
    for k, i in enumerate(streams):
        f = f"[0:v:{i}]"
        steps = ([f"fps={fps}"] if fps else []) + ([f"scale={w}:{h}"] if scale else [])
        chain.append(f"{f}{','.join(steps) or 'null'}[v{k}]")
    if len(streams) == 2:
        chain.append("[v0][v1]hstack=inputs=2[out]")
        out_w, label = w * 2, "[out]"
    else:
        out_w, label = w, "[v0]"
    cmd += ["-filter_complex", ";".join(chain), "-map", label,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    size = out_w * h * 3
    n = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield start + n / out_fps, np.frombuffer(buf, np.uint8).reshape(h, out_w, 3)
            n += 1
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
    if n == 0:
        raise RuntimeError(f"ffmpeg produced no frames for {path}: {' '.join(cmd)}")
