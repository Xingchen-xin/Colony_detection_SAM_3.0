from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import torch
    from transformers import (
        SegformerFeatureExtractor,
        SegformerForSemanticSegmentation,
    )
except Exception:  # pragma: no cover - optional dependency
    torch = None
    SegformerFeatureExtractor = None
    SegformerForSemanticSegmentation = None


class SegFormerSegmenter:
    """Simplified SegFormer-based segmenter."""

    def __init__(self, weights: str, threshold: float = 0.5, device: Optional[str] = None) -> None:
        if torch is None or SegformerFeatureExtractor is None:
            raise ImportError("transformers package with SegFormer is required")
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(weights)
        self.model = SegformerForSemanticSegmentation.from_pretrained(weights)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.threshold = threshold

    def segment(self, image: np.ndarray) -> np.ndarray:
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = outputs.logits.softmax(dim=1)[0, 1]
        mask_np = mask.cpu().numpy()
        return (mask_np > self.threshold).astype(np.uint8)
