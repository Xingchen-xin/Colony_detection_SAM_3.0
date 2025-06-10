"""Utilities for pairing front and back colony results."""
from __future__ import annotations

import time

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
import numpy as np


def load_colony_data(folder: Path) -> List[Dict]:
    """改进的数据加载函数，支持多种数据格式"""
    colonies: List[Dict] = []
    
    if not folder.exists() or not folder.is_dir():
        logging.warning(f"目录不存在或不是目录: {folder}")
        return colonies
    
    # 首先尝试加载 detailed_results.json
    detailed = folder / "results" / "detailed_results.json"
    if detailed.exists():
        try:
            with open(detailed, "r", encoding="utf-8") as f:
                colonies = json.load(f)
            logging.debug(f"从 {detailed} 加载了 {len(colonies)} 个菌落")
            return colonies
        except Exception as e:
            logging.error(f"读取 {detailed} 失败: {e}")
    
    # 尝试加载 analysis_results.csv（主要的结果文件）
    csv_file = folder / "results" / "analysis_results.csv"
    if csv_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            # 转换为字典列表，只保留需要的字段
            for _, row in df.iterrows():
                colony = {
                    'id': row.get('id', ''),
                    'well_position': row.get('well_position', ''),
                    'area': float(row.get('area', 0)),
                    'centroid': eval(row.get('centroid', '(0, 0)')) if pd.notna(row.get('centroid')) else (0, 0),
                    'sam_score': float(row.get('sam_score', 0)),
                    'quality_score': float(row.get('quality_score', 0))
                }
                colonies.append(colony)
            logging.debug(f"从 {csv_file} 加载了 {len(colonies)} 个菌落")
            return colonies
        except Exception as e:
            logging.error(f"读取 {csv_file} 失败: {e}")
    
    # 最后尝试加载单个 colony_*.json 文件
    json_files = list(folder.glob("colony_*.json"))
    if json_files:
        for json_file in sorted(json_files):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    colonies.append(data)
            except Exception as e:
                logging.error(f"读取 {json_file} 失败: {e}")
        logging.debug(f"从单个 JSON 文件加载了 {len(colonies)} 个菌落")
    
    return colonies


# 修复2：改进 save_merged_results，确保文件正确保存

def save_merged_results(path: Path, data: List[Dict]):
    """保存合并的配对结果"""
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / "merged.json"
    
    # 确保数据可序列化
    serializable_data = []
    for item in data:
        try:
            # 处理 numpy 类型
            item_copy = {}
            for k, v in item.items():
                if isinstance(v, np.ndarray):
                    item_copy[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    item_copy[k] = float(v)
                else:
                    item_copy[k] = v
            serializable_data.append(item_copy)
        except Exception as e:
            logging.error(f"序列化数据项失败: {e}")
            serializable_data.append(item)
    
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        logging.info(f"成功保存配对结果到: {out_file}")
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
        front_dir = replicate_dir / "Front"
        back_dir = replicate_dir / "Back"
        front_data = load_colony_data(front_dir)
        back_data = load_colony_data(back_dir)
        # 添加调试日志
        logging.debug(f"Front目录 {front_dir}: 加载了 {len(front_data)} 个菌落")
        logging.debug(f"Back目录 {back_dir}: 加载了 {len(back_data)} 个菌落")

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

def _process_single_replicate(replicate_dir: Path, max_distance: float):
    """处理单个 replicate 的配对"""
    front_dir = replicate_dir / "Front"
    back_dir = replicate_dir / "Back"
    
    # 调试信息
    logging.info(f"处理 replicate: {replicate_dir}")
    logging.info(f"  Front 目录: {front_dir} (存在: {front_dir.exists()})")
    logging.info(f"  Back 目录: {back_dir} (存在: {back_dir.exists()})")
    
    # 加载数据
    front_data = load_colony_data(front_dir)
    back_data = load_colony_data(back_dir)
    
    logging.info(f"  加载 Front 数据: {len(front_data)} 个菌落")
    logging.info(f"  加载 Back 数据: {len(back_data)} 个菌落")
    
    if not front_data and not back_data:
        logging.warning(f"{replicate_dir} 缺少前后视角数据，跳过配对")
        return
    
    # 执行配对
    merged = match_and_merge_colonies(front_data, back_data, max_distance)
    
    # 保存结果
    combined_dir = replicate_dir / "Combined" / "results"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = combined_dir / "merged.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        logging.info(f"  保存配对结果: {output_file} ({len(merged)} 条)")
    except Exception as e:
        logging.error(f"  保存配对结果失败: {e}")
