"""Utilities for pairing front and back colony results."""
from __future__ import annotations

import time

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


def load_colony_data(folder: Path) -> List[Dict]:
    """Load colony data from a result folder."""
    colonies: List[Dict] = []
    if not folder.exists() or not folder.is_dir():
        return colonies

    detailed = folder / "results" / "detailed_results.json"
    if detailed.exists():
        try:
            with open(detailed, "r", encoding="utf-8") as f:
                colonies = json.load(f)
        except Exception as e:
            logging.error(f"读取菌落数据失败: {detailed}: {e}")
        return colonies

    # Fallback to individual colony_*.json files
    for json_file in sorted(folder.glob("colony_*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                colonies.append(data)
        except Exception as e:
            logging.error(f"读取菌落数据失败: {json_file}: {e}")
    return colonies


def save_merged_results(path: Path, data: List[Dict]):
    """Save merged pairing results to path/merged.json."""
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / "merged.json"
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"保存配对结果失败: {out_file}: {e}")


def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def match_and_merge_colonies(
    front_data: List[Dict],
    back_data: List[Dict],
    max_distance: float = 50.0,
) -> List[Dict]:
    """Match colonies from front and back views and merge their data."""

    merged: List[Dict] = []
    used_back = set()

    for f in front_data:
        f_centroid = tuple(f.get("centroid", (0, 0)))
        best_j = None
        best_dist = None
        for j, b in enumerate(back_data):
            if j in used_back:
                continue
            b_centroid = tuple(b.get("centroid", (0, 0)))
            dist = _euclidean_distance(f_centroid, b_centroid)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None and best_dist is not None and best_dist <= max_distance:
            used_back.add(best_j)
            b = back_data[best_j]
            merged.append(
                {
                    "front": f,
                    "back": b,
                    "area": (f.get("area", 0) + b.get("area", 0)) / 2,
                    "centroid_front": f_centroid,
                    "centroid_back": b.get("centroid", (0, 0)),
                    "single_view": False,
                }
            )
        else:
            # no matching back colony
            merged.append({"front": f, "single_view": True})

    for j, b in enumerate(back_data):
        if j not in used_back:
            merged.append({"back": b, "single_view": True})

    return merged


def pair_colonies_across_views(output_dir: str, max_distance: float = 50.0):
    """Pair colonies from front and back results under ``output_dir``."""

    root = Path(output_dir).resolve()

    # If a replicate or orientation directory is provided, handle it directly
    if root.name in {"Front", "Back"}:
        replicate_dir = root.parent
    elif root.name.startswith("replicate_"):
        replicate_dir = root
    else:
        replicate_dir = None

    if replicate_dir:
        front_data = load_colony_data(replicate_dir / "Front")
        back_data = load_colony_data(replicate_dir / "Back")

        if not front_data and not back_data:
            logging.warning(f"{replicate_dir} 缺少前后视角数据，跳过配对")
            return

        if not front_data or not back_data:
            logging.warning(f"{replicate_dir.name} 缺少一侧图像结果")

        merged = match_and_merge_colonies(front_data, back_data, max_distance)
        save_folder = replicate_dir / "Combined" / "results"
        save_merged_results(save_folder, merged)
        logging.info(
            f"配对完成: {replicate_dir.name}, 共 {len(merged)} 条")
        return

    # Otherwise treat as the root output directory and iterate all samples

    start_all = time.time()
    sample_dirs = [d for d in root.iterdir() if d.is_dir()]
    for sample_dir in tqdm(sample_dirs, desc="Pair samples", ncols=80):
        for medium_dir in [d for d in sample_dir.iterdir() if d.is_dir()]:
            for date_dir in [d for d in medium_dir.iterdir() if d.is_dir()]:
                replicate_dirs = [
                    r
                    for r in date_dir.iterdir()
                    if r.is_dir() and r.name.startswith("replicate_")
                ]
                for rep_dir in tqdm(
                    sorted(replicate_dirs),
                    desc=f"{sample_dir.name}-{medium_dir.name}-{date_dir.name}",
                    leave=False,
                    ncols=80,
                ):
                    step_start = time.time()
                    front_dir = rep_dir / "Front"
                    back_dir = rep_dir / "Back"
                    front_data = load_colony_data(front_dir)
                    back_data = load_colony_data(back_dir)

                    if not front_data and not back_data:
                        continue

                    if not front_data or not back_data:
                        logging.warning(
                            f"{sample_dir.name} {medium_dir.name} {rep_dir.name} 缺少一侧图像结果"
                        )

                    merged = match_and_merge_colonies(front_data, back_data, max_distance)

                    save_folder = rep_dir / "Combined" / "results"
                    save_merged_results(save_folder, merged)
                    elapsed = time.time() - step_start
                    logging.info(
                        f"配对完成: {sample_dir.name} {medium_dir.name} {rep_dir.name}, 共 {len(merged)} 条 - {elapsed:.2f}s"
                    )
    total_elapsed = time.time() - start_all
    logging.info(f"配对处理完成，总耗时 {total_elapsed:.2f}s")

