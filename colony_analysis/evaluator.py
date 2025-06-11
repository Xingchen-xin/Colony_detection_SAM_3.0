from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from .config.config_loader import parse_condition_from_path
from .segmenters.sam_segmenter import SamSegmenter
from .segmenters.unet_segmenter import UnetSegmenter

try:
    from colony_analysis.segmenters.fastsam_segmenter import FastSamSegmenter
except Exception:  # pragma: no cover - optional dependency
    FastSamSegmenter = None

try:
    from colony_analysis.segmenters.segformer_segmenter import SegFormerSegmenter
except Exception:  # pragma: no cover - optional dependency
    SegFormerSegmenter = None


def load_model(cfg: Dict[str, Any]):
    name = (cfg.get("name") or cfg.get("model"))
    if not name:
        raise ValueError("Model name missing in config")
    name = name.lower()
    weights = cfg.get("weights")
    if name == "sam":
        return SamSegmenter(model_path=weights, model_type=cfg.get("variant", "vit_b"))
    if name == "unet":
        threshold = float(cfg.get("threshold", 0.5))
        return UnetSegmenter(model_path=weights, threshold=threshold)
    if name == "fastsam":
        if FastSamSegmenter is None:
            raise ImportError("FastSamSegmenter not available")
        return FastSamSegmenter(model_path=weights)
    if name == "segformer":
        if SegFormerSegmenter is None:
            raise ImportError("SegFormerSegmenter not available")
        threshold = float(cfg.get("threshold", 0.5))
        return SegFormerSegmenter(weights=weights, threshold=threshold)
    raise ValueError(f"Unsupported model: {name}")


def compute_metrics(pred_masks: List[np.ndarray], true_masks: List[np.ndarray]) -> Dict[str, float]:
    tp = min(len(pred_masks), len(true_masks))
    fp = max(0, len(pred_masks) - len(true_masks))
    fn = max(0, len(true_masks) - len(pred_masks))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive": fp,
        "false_negative": fn,
    }


def evaluate_colonies(image_path: str, config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    medium, side = parse_condition_from_path(image_path)
    key = f"{medium}_{side}"
    default_cfg = (config.get("defaults", {}) or {}).get(key)

    experiment_cfgs: List[Dict[str, Any]] = []
    for exp in config.get("experiments", []):
        if exp.get("medium") == medium and exp.get("side") == side:
            models = exp.get("models", [])
            for m in models:
                if isinstance(m.get("threshold"), list):
                    for t in m["threshold"]:
                        mc = m.copy()
                        mc["threshold"] = t
                        experiment_cfgs.append(mc)
                else:
                    experiment_cfgs.append(m)

    if not experiment_cfgs and default_cfg:
        experiment_cfgs.append(default_cfg)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    gt_masks: List[np.ndarray] = []
    base = Path(image_path).stem
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    results = []
    for cfg in experiment_cfgs:
        try:
            model = load_model(cfg)
            pred = model.segment(image)
            if isinstance(pred, np.ndarray):
                pred_masks = [pred]
            else:
                pred_masks = pred
            metrics = compute_metrics(pred_masks, gt_masks)
        except Exception as e:  # pragma: no cover - runtime safety
            metrics = {"error": str(e)}
        result = {**cfg, **metrics}
        results.append(result)
        with open(log_dir / f"{base}_{cfg.get('name','model')}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    best = max((r for r in results if "f1" in r), key=lambda x: x.get("f1", 0.0), default=None)
    return {"best_model": best.get("name") if best else None, "metrics": results}
