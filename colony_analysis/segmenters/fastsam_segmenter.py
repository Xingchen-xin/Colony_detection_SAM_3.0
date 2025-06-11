from __future__ import annotations

from typing import List, Optional
import numpy as np

try:
    from ultralytics import FastSAM
except Exception:  # pragma: no cover - optional dependency
    FastSAM = None


class FastSamSegmenter:
    """Lightweight wrapper for FastSAM segmentation."""

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        if FastSAM is None:
            raise ImportError("ultralytics package is required for FastSAM")
        self.model = FastSAM(model_path)
        self.device = device

    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        """Return list of masks predicted by FastSAM."""
        results = self.model.predict(source=image, device=self.device or "cpu")
        masks = getattr(results, "masks", None)
        if masks is None:
            return []
        return [m for m in masks.data]
