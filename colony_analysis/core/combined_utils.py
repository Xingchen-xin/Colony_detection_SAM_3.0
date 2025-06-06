"""Utility functions for combining Front and Back metrics."""

from pathlib import Path
from typing import Dict


def _load_stats(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not path.exists():
        return metrics
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                # 非数字数值忽略
                pass
    return metrics


def combine_metrics(front_stats_path: str, back_stats_path: str) -> Dict[str, float]:
    """读取前后两个统计文件, 合并为统一指標字典."""
    front_metrics = {f"front_{k}": v for k, v in _load_stats(Path(front_stats_path)).items()}
    back_metrics = {f"back_{k}": v for k, v in _load_stats(Path(back_stats_path)).items()}
    combined = {**front_metrics, **back_metrics}
    # TODO: 在此处根据需要新增聚合指标
    return combined
