"""Zero-shot detector: text prompts in, boxes out. No training data needed."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

PROMPTS = ["a crocodile", "an alligator", "a baby crocodile", "a crocodile head in water",
           "a large lizard"]

MODELS = {
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "owlv2-large": "google/owlv2-large-patch14-ensemble",
    "gdino": "IDEA-Research/grounding-dino-base",
    "gdino-tiny": "IDEA-Research/grounding-dino-tiny",
}


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    label: str


class Detector:
    def __init__(self, model: str = "owlv2", device: str | None = None,
                 prompts: list[str] | None = None, threshold: float = 0.1):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.name = MODELS.get(model, model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompts = prompts or PROMPTS
        self.threshold = threshold
        self.grounding = "grounding" in self.name.lower()
        self.proc = AutoProcessor.from_pretrained(self.name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.name, torch_dtype=dtype).to(self.device).eval()
        self.dtype = dtype

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image]) -> list[list[Box]]:
        sizes = [(im.height, im.width) for im in images]
        if self.grounding:
            # Grounding DINO wants one lower-case string of phrases separated by " . "
            text = [" . ".join(p.lower() for p in self.prompts) + " ."] * len(images)
        else:
            text = [self.prompts] * len(images)
        inputs = self.proc(images=images, text=text, return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                  for k, v in inputs.items()}
        out = self.model(**inputs)
        if self.grounding:
            res = self.proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], threshold=self.threshold,
                text_threshold=self.threshold, target_sizes=sizes)
        else:
            # OWLv2 pads the image to a square before resizing, so boxes are
            # relative to the padded square, not the original image.
            sq = [(max(s), max(s)) for s in sizes]
            res = self.proc.post_process_grounded_object_detection(
                out, threshold=self.threshold, target_sizes=sq, text_labels=text)
        boxes = []
        for r, (h, w) in zip(res, sizes):
            labels = r.get("text_labels", r.get("labels"))
            per = []
            for b, s, l in zip(r["boxes"].float().tolist(), r["scores"].float().tolist(), labels):
                x0, y0, x1, y1 = b
                x0, x1 = max(0.0, x0), min(float(w), x1)
                y0, y1 = max(0.0, y0), min(float(h), y1)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                per.append(Box(x0, y0, x1, y1, s, l if isinstance(l, str) else str(l)))
            boxes.append(nms(per))
        return boxes


def iou(a: Box, b: Box) -> float:
    iw = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    ih = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = iw * ih
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list[Box], thr: float = 0.5) -> list[Box]:
    """Keep the best-scoring box among overlapping ones (prompts overlap a lot)."""
    keep: list[Box] = []
    for b in sorted(boxes, key=lambda b: -b.score):
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep
