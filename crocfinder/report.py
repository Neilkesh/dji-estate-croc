"""Write the review gallery: candidates.csv + index.html next to a crops/ folder."""
from __future__ import annotations

import csv
import html
from pathlib import Path

FIELDS = ["rank", "video", "t", "timecode", "yaw", "pitch", "label", "det_score",
          "text_score", "ref_score", "persist", "rank_score", "box_px", "crop", "context"]


def timecode(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:05.2f}"


def rank_score(c: dict) -> float:
    """Single number to sort by. Detector and verifier must both agree to rank high."""
    persist = min(c.get("persist", 0), 3) / 3
    return (c["det_score"] * (0.25 + c.get("text_score", 0.0))
            * (1 + 0.5 * persist) * (1 + max(0.0, c.get("ref_score", 0.0))))


def write(out_dir: Path, video: str, cands: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cands:
        c["video"] = video
        c["timecode"] = timecode(c["t"])
        c["rank_score"] = rank_score(c)
    cands.sort(key=lambda c: -c["rank_score"])
    for i, c in enumerate(cands, 1):
        c["rank"] = i
    with open(out_dir / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        for c in cands:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})

    cards = []
    for c in cands:
        cards.append(f"""<figure>
<a href="{html.escape(c['context'])}"><img loading="lazy" src="{html.escape(c['crop'])}"></a>
<figcaption>#{c['rank']} &middot; <b>{c['timecode']}</b> &middot; {html.escape(c['label'])}<br>
score {c['rank_score']:.3f} (det {c['det_score']:.2f}, text {c.get('text_score', 0):.2f},
ref {c.get('ref_score', 0):.2f}, seen in {c.get('persist', 0)} nearby frames)<br>
looking yaw {c['yaw']:.0f}&deg;, pitch {c['pitch']:.0f}&deg; &middot; {c['box_px']} px</figcaption>
</figure>""")
    info = " &middot; ".join(f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in meta.items())
    (out_dir / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>Croc candidates - {html.escape(video)}</title>
<style>
body{{font:14px system-ui,sans-serif;margin:16px;background:#111;color:#ddd}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
figure{{margin:0;background:#1c1c1c;border-radius:6px;overflow:hidden}}
img{{width:100%;display:block}} figcaption{{padding:8px;line-height:1.4}}
p{{color:#999}} a{{color:#8cf}}
</style>
<h1>{html.escape(video)}: {len(cands)} candidates</h1>
<p>{info}</p>
<p>Best candidates first. Click a crop to see the full tile it came from.
Timecode is the position in the original video.</p>
<main>{''.join(cards)}</main>""")
