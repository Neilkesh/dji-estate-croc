"""Second opinion on each candidate crop, plus a persistence check across frames."""
from __future__ import annotations

import torch
from PIL import Image

from .geometry import angular_distance

POSITIVE = ["a photo of a crocodile", "an aerial photo of a crocodile on a river bank",
            "a crocodile floating in water seen from above", "a baby crocodile"]
NEGATIVE = ["a log floating in water", "a rock", "mud", "grass", "a tree branch",
            "water", "a bird", "a boat", "a cow", "a person", "a shadow"]


class Verifier:
    """Scores a crop two ways with one image-text model:

    text_score: probability mass on croc captions vs look-alike captions
    ref_score:  best cosine similarity to reference croc photos (the phone photo etc.)
    """

    def __init__(self, refs: list[Image.Image] | None = None,
                 model: str = "google/siglip-base-patch16-224", device: str | None = None):
        from transformers import AutoModel, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()
        with torch.inference_mode():
            t = self.proc(text=POSITIVE + NEGATIVE, padding="max_length", return_tensors="pt")
            self.text = self._norm(self._feat(self.model.get_text_features(**t.to(self.device))))
        self.refs = self.embed(refs) if refs else None

    @staticmethod
    def _feat(x):
        return x if isinstance(x, torch.Tensor) else x.pooler_output

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> torch.Tensor:
        x = self.proc(images=images, return_tensors="pt").to(self.device)
        return self._norm(self._feat(self.model.get_image_features(**x)))

    @torch.inference_mode()
    def score(self, crops: list[Image.Image]) -> list[tuple[float, float]]:
        if not crops:
            return []
        e = self.embed(crops)
        probs = (e @ self.text.T * 100).softmax(-1)
        text_score = probs[:, :len(POSITIVE)].sum(-1)
        if self.refs is not None:
            ref_score = (e @ self.refs.T).max(-1).values
        else:
            ref_score = torch.zeros(len(crops), device=e.device)
        return list(zip(text_score.tolist(), ref_score.tolist()))


def mark_persistent(cands: list[dict], max_dt: float, max_deg: float = 12.0) -> None:
    """Set c['persist'] = number of other sampled frames within max_dt seconds
    that hold a candidate in roughly the same direction. A real animal shows up
    in several consecutive frames; one-frame flickers usually do not.
    """
    cands.sort(key=lambda c: c["t"])
    for i, c in enumerate(cands):
        seen = set()
        for j in range(i - 1, -1, -1):
            o = cands[j]
            if c["t"] - o["t"] > max_dt:
                break
            if o["t"] != c["t"] and angular_distance((c["yaw"], c["pitch"]), (o["yaw"], o["pitch"])) < max_deg:
                seen.add(o["t"])
        for j in range(i + 1, len(cands)):
            o = cands[j]
            if o["t"] - c["t"] > max_dt:
                break
            if o["t"] != c["t"] and angular_distance((c["yaw"], c["pitch"]), (o["yaw"], o["pitch"])) < max_deg:
                seen.add(o["t"])
        c["persist"] = len(seen)
