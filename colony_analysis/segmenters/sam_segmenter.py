import os
from typing import Tuple, List
import numpy as np

from ..core.sam_model import SAMModel


class SamSegmenter:
    """Wrapper around :class:`SAMModel` providing a simple interface for the
    pipeline."""

    def __init__(self, model_path: str = None, model_type: str = "vit_b") -> None:
        # ``SAMModel`` resolves a default checkpoint path if ``model_path`` is None
        self.model = SAMModel(model_type=model_type, checkpoint_path=model_path)
        # Expose the underlying mask generator for direct use
        self.mask_generator = self.model.mask_generator

    def segment(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """Segment the image using SAM automatic mask generation."""
        masks = self.mask_generator.generate(image)
        mask_list = [m["segmentation"] for m in masks]
        scores = [m.get("stability_score", 1.0) for m in masks]
        return mask_list, scores
