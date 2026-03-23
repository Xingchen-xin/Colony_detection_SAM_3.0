"""YOLO + SAM two-stage segmenter for Streptomyces colony detection.

Stage 1 — YOLO (ultralytics): fast bounding-box detection of candidate colonies.
Stage 2 — SAM: precise instance segmentation prompted by YOLO bboxes.

This is typically 5-10x faster than SamAutomaticMaskGenerator because SAM only
runs the lightweight mask decoder (not the full grid search) for each bbox prompt.

Fallback: if YOLO detects nothing, the module can optionally fall back to the
existing SamAutomaticMaskGenerator path.

Usage:
    segmenter = YoloSamSegmenter(
        yolo_weights="models/colony_yolo.pt",
        sam_checkpoint="models/sam_vit_b_01ec64.pth",
    )
    masks, scores, bboxes = segmenter.segment(image_rgb)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YOLO detector wrapper
# ---------------------------------------------------------------------------

class YoloDetector:
    """Lightweight wrapper around ultralytics YOLO for colony bbox detection."""

    def __init__(
        self,
        weights: str | Path = "models/colony_yolo.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        imgsz: int = 1280,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO detection. "
                "Install with: pip install ultralytics"
            )
        self.model = YOLO(str(weights))
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.device = device
        self.imgsz = imgsz
        logger.info(f"YoloDetector initialized: weights={weights}, conf={conf_threshold}")

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO detection, return list of {bbox, conf, cls}.

        bbox format: [x1, y1, x2, y2] in pixel coordinates.
        """
        results = self.model.predict(
            image_rgb,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device if self.device != "auto" else None,
            verbose=False,
        )
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(boxes.conf[i].cpu()),
                    "class": int(boxes.cls[i].cpu()) if boxes.cls is not None else 0,
                })
        logger.info(f"YOLO detected {len(detections)} candidate colonies")
        return detections


# ---------------------------------------------------------------------------
# SAM refiner (bbox-prompted)
# ---------------------------------------------------------------------------

class SamRefiner:
    """Use SAM's SamPredictor to refine YOLO bboxes into precise masks."""

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: str | Path = "models/sam_vit_b_01ec64.pth",
        device: Optional[str] = None,
    ):
        import torch
        from segment_anything import sam_model_registry, SamPredictor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        sam = sam_model_registry[model_type]()
        state = torch.load(str(checkpoint), map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        sam.load_state_dict(state, strict=False)
        sam.to(self.device)

        self.predictor = SamPredictor(sam)
        self._image_set = False
        logger.info(f"SamRefiner initialized: {model_type} on {self.device}")

    def set_image(self, image_rgb: np.ndarray):
        """Encode the image once (expensive), then reuse for all bbox prompts."""
        self.predictor.set_image(image_rgb)
        self._image_set = True

    def refine_bbox(self, bbox: List[int]) -> Tuple[np.ndarray, float]:
        """Given a bbox [x1,y1,x2,y2], produce a refined binary mask + score."""
        if not self._image_set:
            raise RuntimeError("Call set_image() before refine_bbox()")

        input_box = np.array(bbox)
        masks, scores, _ = self.predictor.predict(
            box=input_box[None, :],
            multimask_output=True,
        )
        # Pick the mask with highest predicted IoU
        best_idx = np.argmax(scores)
        return masks[best_idx], float(scores[best_idx])


# ---------------------------------------------------------------------------
# Combined YOLO+SAM segmenter
# ---------------------------------------------------------------------------

class YoloSamSegmenter:
    """Two-stage segmenter: YOLO bbox → SAM mask refinement.

    Conforms to the same interface as SamSegmenter/FastSamSegmenter
    so it can be used as a drop-in in the evaluator pipeline.
    """

    def __init__(
        self,
        yolo_weights: str | Path = "models/colony_yolo.pt",
        sam_checkpoint: str | Path = "models/sam_vit_b_01ec64.pth",
        sam_model_type: str = "vit_b",
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        yolo_imgsz: int = 1280,
        device: Optional[str] = None,
        fallback_to_auto: bool = True,
        min_mask_area: int = 200,
    ):
        self.yolo = YoloDetector(
            weights=yolo_weights,
            conf_threshold=yolo_conf,
            iou_threshold=yolo_iou,
            device=device or "auto",
            imgsz=yolo_imgsz,
        )
        self.sam = SamRefiner(
            model_type=sam_model_type,
            checkpoint=sam_checkpoint,
            device=device,
        )
        self.fallback_to_auto = fallback_to_auto
        self.min_mask_area = min_mask_area
        self._last_detections: List[Dict] = []

    def segment(
        self, image_rgb: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[List[int]]]:
        """Run full YOLO+SAM pipeline.

        Returns:
            masks: list of binary masks (H, W)
            scores: list of quality scores
            bboxes: list of [x1, y1, x2, y2] bounding boxes
        """
        # Stage 1: YOLO detection
        detections = self.yolo.detect(image_rgb)
        self._last_detections = detections

        if not detections and self.fallback_to_auto:
            logger.warning("YOLO found no colonies — falling back to SAM auto mode")
            return self._fallback_sam_auto(image_rgb)

        # Stage 2: SAM refinement
        self.sam.set_image(image_rgb)  # encode once

        masks = []
        scores = []
        bboxes = []

        for det in detections:
            bbox = det["bbox"]
            mask, score = self.sam.refine_bbox(bbox)

            # Filter tiny artifacts
            if np.sum(mask) < self.min_mask_area:
                continue

            masks.append(mask)
            # Combined score: YOLO confidence × SAM IoU score
            combined = det["confidence"] * score
            scores.append(combined)
            bboxes.append(bbox)

        logger.info(
            f"YOLO+SAM: {len(detections)} detections → {len(masks)} valid masks"
        )
        return masks, scores, bboxes

    def _fallback_sam_auto(
        self, image_rgb: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[List[int]]]:
        """Fallback to SAM automatic mask generation if YOLO finds nothing."""
        from segment_anything import SamAutomaticMaskGenerator

        generator = SamAutomaticMaskGenerator(
            self.sam.predictor.model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=self.min_mask_area,
        )
        results = generator.generate(image_rgb)

        masks = [r["segmentation"] for r in results]
        scores = [r["predicted_iou"] for r in results]
        bboxes = [
            [int(r["bbox"][0]), int(r["bbox"][1]),
             int(r["bbox"][0] + r["bbox"][2]), int(r["bbox"][1] + r["bbox"][3])]
            for r in results
        ]
        return masks, scores, bboxes

    # ── Convenience methods for pipeline integration ──

    def get_detections(self) -> List[Dict]:
        """Return raw YOLO detections from last run."""
        return self._last_detections

    def segment_everything(
        self, image_rgb: np.ndarray, return_logits: bool = False
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Compatible interface with existing SAMModel.segment_everything()."""
        masks, scores, _ = self.segment(image_rgb)
        return masks, scores
