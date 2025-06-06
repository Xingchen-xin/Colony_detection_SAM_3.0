"""Utilities for pairing front and back colony results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple


def load_colony_data(folder: Path) -> List[Dict]:
    """Load colony_*.json files from the given folder."""
    colonies: List[Dict] = []
    if not folder.exists() or not folder.is_dir():
        return colonies

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
    """Pair colonies from front and back results under the given output directory."""

    root = Path(output_dir).resolve()
    # If called with orientation/replicate path, strip to root
    if root.parent.name in {"Front", "Back"} and root.name.startswith("replicate_"):
        root = root.parents[4]

    for sample_dir in root.iterdir():
        if not sample_dir.is_dir():
            continue
        for medium_dir in sample_dir.iterdir():
            if not medium_dir.is_dir():
                continue
            front_root = medium_dir / "Front"
            back_root = medium_dir / "Back"
            replicates = set()
            if front_root.exists():
                replicates.update(p.name for p in front_root.iterdir() if p.is_dir())
            if back_root.exists():
                replicates.update(p.name for p in back_root.iterdir() if p.is_dir())

            for rep in replicates:
                front_dir = front_root / rep
                back_dir = back_root / rep
                front_data = load_colony_data(front_dir)
                back_data = load_colony_data(back_dir)

                if not front_data and not back_data:
                    continue

                if not front_data or not back_data:
                    logging.warning(
                        f"{sample_dir.name} {medium_dir.name} {rep} 缺少一侧图像结果"
                    )

                merged = match_and_merge_colonies(front_data, back_data, max_distance)

                save_folder = medium_dir / "paired" / rep
                save_merged_results(save_folder, merged)
                logging.info(
                    f"配对完成: {sample_dir.name} {medium_dir.name} {rep}, 共 {len(merged)} 条"
                )

