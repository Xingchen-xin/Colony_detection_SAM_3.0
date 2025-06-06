# ============================================================================
# 3. colony_analysis/pipeline.py - 分析管道
# ============================================================================

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

import cv2

from .analysis import ColonyAnalyzer
from .config import ConfigManager
from .core import ColonyDetector, SAMModel, combine_metrics
from .utils import (
    ImageValidator,
    ResultManager,
    Visualizer,
    collect_all_images,
    parse_filename,
)


class AnalysisPipeline:
    """分析管道 - 协调整个分析流程"""

    def __init__(self, args):
        """初始化分析管道"""
        self.args = args
        self.start_time = None
        self.config = None
        self.sam_model = None
        self.detector = None
        self.analyzer = None
        self.result_manager = None

    def run(self):
        """运行完整的分析流程"""
        self.start_time = time.time()
        try:
            # 1. 初始化组件
            self._initialize_components()

            # 2. 加载和验证图像
            img_rgb = self._load_and_validate_image()

            # -------- 新增 --------
            # 创建 96-孔网格并存到 config，避免后面找不到
            if not hasattr(self.config, "plate_grid"):
                self.config.plate_grid = self.detector._create_plate_grid(
                    img_rgb.shape[:2]
                )
            # -------- 新增结束 ------

            # 3. 执行检测
            if getattr(self.args, "force_96plate_detection", False):
                # 强制96孔板检测模式
                colonies = self._force_96plate_detection(img_rgb)
            else:
                colonies = self._detect_colonies(img_rgb)

            # —— 如果是 hybrid 模式，检测器内部已经将每个 colony 对应到 well_position，
            #    此处额外汇总哪些孔位漏检 ——
            if self.args.mode == "hybrid":
                plate_wells = set(self.config.plate_grid.keys())
                detected_wells = {
                    c["well_position"]
                    for c in colonies
                    if c.get("well_position", "").upper() in plate_wells
                }
                missing = plate_wells - detected_wells
                if missing:
                    logging.warning(f"以下孔位未检测到任何菌落：{sorted(missing)}")

            # 4. 如果未检测到菌落且开启调试，则保存原始 SAM 掩码用于排查
            if not colonies and self.args.debug:
                self.detector.save_raw_debug(img_rgb)
                return {
                    "total_colonies": 0,
                    "elapsed_time": time.time() - self.start_time,
                    "output_dir": self.args.output,
                    "mode": self.args.mode,
                    "model": self.args.model,
                    "advanced": self.args.advanced,
                }

            # 5. 执行分析
            analyzed_colonies = self._analyze_colonies(colonies)

            # 6. 离群值检测
            if getattr(self.args, "outlier_detection", False):
                try:
                    import pandas as pd
                    from colony_analysis.outlier import detect_outliers
                    df = pd.DataFrame(analyzed_colonies)
                    metric = getattr(self.args, "outlier_metric", "area")
                    threshold = getattr(self.args, "outlier_threshold", 3.0)
                    df_out = detect_outliers(df, metric=metric, threshold=threshold)
                    # 更新 analyzed_colonies
                    analyzed_colonies = df_out.to_dict(orient="records")
                except Exception as e:
                    logging.error(f"离群值检测失败: {e}")

            # 7. 保存结果
            self._save_results(analyzed_colonies, img_rgb)

            # 8. 返回结果摘要
            return self._generate_summary(analyzed_colonies)

        except Exception as e:
            logging.error(f"分析管道执行失败: {e}")
            raise

    def _force_96plate_detection(self, img_rgb):
        """
        强制96孔板检测：对每个预设孔位区域内查找候选菌落，输出96个条目（无菌落填推测/补全信息）
        新增支持推测未生长菌落的可视化与数据补全。
        """
        import numpy as np
        from copy import deepcopy
        import cv2
        from PIL import Image, ImageDraw, ImageFont

        # 获取plate_grid: well_id -> (row, col, cx, cy, r)
        plate_grid = self.config.plate_grid
        # 1. 检测所有菌落
        colonies = self.detector.detect(img_rgb, mode="grid")
        # 2. 分配菌落到最近well_id
        well_to_candidates = {well: [] for well in plate_grid}
        for c in colonies:
            centroid = c.get("centroid", None)
            if centroid is None:
                continue
            cy, cx = centroid
            # 找到最近的well
            min_dist = float("inf")
            min_well = None
            for well_id, (row, col, wx, wy, wr) in plate_grid.items():
                d = np.hypot(cx - wx, cy - wy)
                if d < min_dist and d < wr * 1.5:  # 允许一定范围
                    min_dist = d
                    min_well = well_id
            if min_well is not None:
                well_to_candidates[min_well].append(c)
        # 3. 统计所有已检测菌落的半径
        detected_colonies = []
        for candlist in well_to_candidates.values():
            detected_colonies.extend(candlist)
        # 取所有菌落的半径
        radii = []
        for c in detected_colonies:
            if "radius" in c:
                radii.append(c["radius"])
            elif "mask" in c:
                # 估算半径
                area = np.sum(c["mask"] > 0)
                radii.append(np.sqrt(area / np.pi))
        median_radius = np.median(radii) if radii else 25  # default_radius
        # 4. 遍历每个孔位，输出菌落或推测未生长
        forced_colonies = []
        fallback_policy = getattr(self.args, "fallback_null_policy", "fill")
        # 用于可视化的输出图像副本
        vis_img = img_rgb.copy()
        # 用于PIL标注
        pil_img = Image.fromarray(vis_img)
        draw = ImageDraw.Draw(pil_img)
        # 尝试加载字体
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font = None
        for well_id in sorted(plate_grid.keys()):
            candidates = well_to_candidates[well_id]
            # 获取理论孔位中心与半径
            row, col, wx, wy, wr = plate_grid[well_id]
            wx, wy = int(wx), int(wy)
            est_radius = int(median_radius)
            est_area = float(np.pi * (est_radius ** 2))
            if candidates:
                # 按分数或面积排序，选取最佳
                best = max(candidates, key=lambda cc: cc.get("scores", {}).get("overall_score", 0) if "scores" in cc else cc.get("area", 0))
                best = deepcopy(best)
                best["well_position"] = well_id
                best["forced_96plate"] = True
                # 标记状态
                best["colony_status"] = "detected"
                best["growth_status"] = "normal"
                forced_colonies.append(best)
            else:
                if fallback_policy == "skip":
                    continue
                # 提取局部区域特征
                # 提取一个圆区域的mask
                mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (wx, wy), est_radius, 1, thickness=-1)
                # 提取局部灰度均值
                local_pixels = img_rgb[mask > 0]
                if len(local_pixels.shape) == 3 and local_pixels.shape[1] == 3:
                    # 转灰度
                    local_gray = np.mean(local_pixels, axis=1)
                else:
                    local_gray = local_pixels
                local_mean = float(np.mean(local_gray)) if local_gray.size > 0 else 0.0
                # dot_count: 统计区域内亮点数（可自定义，此处用亮度高于阈值的像素数/100）
                dot_count = int(np.sum(local_gray > 180) // 100)
                # 判断是否为non-growing
                area_thresh = est_area * 0.3
                intensity_thresh = 120
                growth_status = "non-growing" if (est_area < area_thresh or local_mean < intensity_thresh) else "normal"
                # 构造推测菌落条目
                inferred_colony = {
                    "id": f"colony_{well_id}",
                    "well_id": well_id,
                    "well_position": well_id,
                    "center": (wx, wy),
                    "radius": est_radius,
                    "area": est_area,
                    "intensity_mean": local_mean,
                    "dot_count": dot_count,
                    "colony_status": "inferred",
                    "growth_status": growth_status,
                    "forced_96plate": True,
                    "is_null": True,
                    "features": {
                        "area": est_area,
                        "intensity_mean": local_mean,
                        "radius": est_radius,
                        "dot_count": dot_count,
                    },
                    "scores": {},
                    "phenotype": {},
                }
                forced_colonies.append(inferred_colony)
                # --- 可视化: 灰色虚线圆, 标记“推测未生长” ---
                circle_color = (160, 160, 160, 180)  # 灰色
                thickness = 2
                # 绘制虚线圆
                dash_len = 6
                for angle in range(0, 360, dash_len * 2):
                    start_angle = angle
                    end_angle = angle + dash_len
                    draw.arc([wx - est_radius, wy - est_radius, wx + est_radius, wy + est_radius],
                             start=start_angle, end=end_angle, fill=circle_color, width=thickness)
                # 标注文字
                label = "推测未生长"
                text_color = (220, 0, 0) if growth_status == "non-growing" else (80, 80, 80)
                text_xy = (wx + est_radius + 2, wy - 12)
                if font:
                    draw.text(text_xy, label, fill=text_color, font=font)
                else:
                    draw.text(text_xy, label, fill=text_color)
        # 保存可视化到debug目录
        debug_dir = self.result_manager.directories.get("debug", None)
        if debug_dir:
            vis_save_path = os.path.join(debug_dir, "force96_inferred.png")
            try:
                pil_img.save(vis_save_path)
            except Exception:
                pass
        return forced_colonies

    def _initialize_components(self):
        """初始化所有组件"""
        logging.info("初始化组件...")

        # 配置管理器
        self.config = ConfigManager(self.args.config)
        self.config.update_from_args(self.args)
        self._apply_medium_specific_config()

        # SAM模型
        self.sam_model = SAMModel(model_type=self.args.model, config=self.config)

        # 结果管理器
        self.result_manager = ResultManager(self.args.output)

        # 检测器
        self.detector = ColonyDetector(
            sam_model=self.sam_model,
            config=self.config,
            result_manager=self.result_manager,
            debug=self.args.debug,
        )

        # 分析器
        self.analyzer = ColonyAnalyzer(
            sam_model=self.sam_model,
            config=self.config,
            debug=self.args.debug,
            orientation=getattr(self.args, "orientation", "front"),
        )

        logging.info("组件初始化完成")

    def _apply_medium_specific_config(self):
        """根据培养基调整参数（示例占位实现）"""
        medium = getattr(self.args, "medium", "").lower()
        if medium == "r5":
            # R5 培养基适当提高最小菌落面积阈值
            self.config.detection.min_colony_area = max(
                1500, self.config.detection.min_colony_area
            )
        elif medium == "mmm":
            # MMM 培养基可能菌落更小
            self.config.detection.min_colony_area = max(
                1000, self.config.detection.min_colony_area // 2
            )

    def _load_and_validate_image(self):
        """加载和验证图像"""
        logging.info(f"加载图像: {self.args.image}")

        # 检查文件是否存在
        if not Path(self.args.image).exists():
            raise FileNotFoundError(f"图像文件不存在: {self.args.image}")

        # 加载图像
        img = cv2.imread(self.args.image)
        if img is None:
            raise ValueError(f"无法读取图像文件: {self.args.image}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 验证图像
        is_valid, error_msg = ImageValidator.validate_image(img_rgb)
        if not is_valid:
            raise ValueError(f"图像验证失败: {error_msg}")

        logging.info(f"图像加载成功，尺寸: {img_rgb.shape}")
        return img_rgb

    def _detect_colonies(self, img_rgb):
        """执行菌落检测"""
        logging.info("开始菌落检测...")

        colonies = self.detector.detect(img_rgb, mode=self.args.mode)

        if not colonies:
            logging.warning("未检测到任何菌落，将继续生成调试信息")
            return []

        logging.info(f"检测到 {len(colonies)} 个菌落")

        # ======== 自动重命名 unmapped 的 Debug 图为对应的孔位标签 ========
        debug_dir = Path(self.args.output) / "debug"
        for colony in colonies:
            original_id = colony.get("id", "")
            well_id = colony.get("well_position", "")
            if (
                original_id.startswith("colony_")
                and well_id
                and not well_id.startswith("unmapped")
            ):
                idx = original_id.split("_")[1]
                # 重命名自动检测阶段的调试图
                old_name = f"debug_colony_unmapped_{idx}.png"
                new_name = f"debug_colony_{well_id}_{idx}.png"
                old_path = debug_dir / old_name
                new_path = debug_dir / new_name
                if old_path.exists():
                    os.rename(str(old_path), str(new_path))
                # 重命名 raw SAM 掩码调试图
                old_raw = f"debug_raw_mask_unmapped_{idx}.png"
                new_raw = f"debug_raw_mask_{well_id}_{idx}.png"
                old_raw_path = debug_dir / old_raw
                new_raw_path = debug_dir / new_raw
                if old_raw_path.exists():
                    os.rename(str(old_raw_path), str(new_raw_path))
        # ======== unmapped Debug 重命名结束 ========

        # ======== 同样重命名蓝/红 debug 代谢图，匹配到相应孔位 ========
        # 假设 blue/red 图保存在 debug_metabolite/ 目录下，文件名格式为 blue_{cy}_{cx}.png 或 red_{cy}_{cx}.png
        metabolite_debug_dir = Path(self.args.output) / "debug_metabolite"
        if metabolite_debug_dir and metabolite_debug_dir.exists():
            for colony in colonies:
                well_id = colony.get("well_position", "")
                if well_id and not well_id.startswith("unmapped"):
                    # 获取质心坐标
                    centroid = colony.get("centroid", None)
                    if centroid:
                        cy, cx = int(centroid[0]), int(centroid[1])
                        # 蓝色代谢图
                        old_blue = f"blue_{cy}_{cx}.png"
                        new_blue = f"blue_{well_id}_{cy}_{cx}.png"
                        old_blue_path = metabolite_debug_dir / old_blue
                        new_blue_path = metabolite_debug_dir / new_blue
                        if old_blue_path.exists():
                            os.rename(str(old_blue_path), str(new_blue_path))
                        # 红色代谢图
                        old_red = f"red_{cy}_{cx}.png"
                        new_red = f"red_{well_id}_{cy}_{cx}.png"
                        old_red_path = metabolite_debug_dir / old_red
                        new_red_path = metabolite_debug_dir / new_red
                        if old_red_path.exists():
                            os.rename(str(old_red_path), str(new_red_path))
        # ======== 蓝/红 debug 重命名结束 ========

        return colonies

    def _analyze_colonies(self, colonies):
        """执行菌落分析"""
        logging.info("开始菌落分析...")

        analyzed_colonies = self.analyzer.analyze(colonies, advanced=self.args.advanced)

        logging.info(f"分析完成，共 {len(analyzed_colonies)} 个菌落")
        return analyzed_colonies

    def _save_results(self, analyzed_colonies, img_rgb):
        """保存结果"""
        logging.info("保存分析结果...")

        # 保存基本结果
        self.result_manager.save_all_results(analyzed_colonies, self.args)

        # 生成可视化
        if self.args.debug:
            visualizer = Visualizer(self.args.output)
            visualizer.create_debug_visualizations(img_rgb, analyzed_colonies)

        # 额外: force_96plate_detection 推测未生长可视化已在 _force_96plate_detection 内保存

        logging.info(f"结果已保存到: {self.args.output}")

    def _generate_summary(self, analyzed_colonies):
        """生成结果摘要"""
        elapsed_time = time.time() - self.start_time

        return {
            "total_colonies": len(analyzed_colonies),
            "elapsed_time": elapsed_time,
            "output_dir": self.args.output,
            "mode": self.args.mode,
            "model": self.args.model,
            "advanced": self.args.advanced,
        }


def batch_medium_pipeline(input_folder: str, output_folder: str):
    """批量处理多培养基、多角度、多重复的图像"""
    img_paths = collect_all_images(input_folder)
    if not img_paths:
        logging.warning(f"在 {input_folder} 未发现任何图片文件。")
        return

    from collections import defaultdict
    groups: Dict[Tuple[str, str, str], Dict[str, Path]] = defaultdict(dict)

    for img_path in img_paths:
        sample_name, medium, orientation, replicate = parse_filename(img_path.stem)
        key = (sample_name, medium, replicate)
        groups[key][orientation] = img_path

    # 新增: 自动解析文件名并构造输出路径
    from pathlib import Path
    import re

    def get_output_path(base_output_dir, image_path, replicate_id, view_type):
        # 解析如：Lib96_Ctrl_@MMM_Back20250401_09191796
        filename = Path(image_path).stem
        match = re.match(r"(?P<group>Lib96_\w+)_@(?P<medium>\w+)_(?P<view>Back|Front)(?P<dateid>\d+)", filename)
        if not match:
            raise ValueError(f"Unexpected filename format: {filename}")

        group = match.group("group")
        medium = match.group("medium")
        dateid = match.group("dateid")

        out_dir = Path(base_output_dir) / group / medium / dateid / f"replicate_{replicate_id}" / view_type
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    summary_data: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)

    start_all = time.time()
    group_items = list(groups.items())
    for (sample_name, medium, replicate), ori_dict in tqdm(
        group_items, desc="Batch processing", ncols=80
    ):
        step_start = time.time()

        try:
            # --- Use actual AnalysisPipeline for each orientation ---
            from argparse import Namespace
            from .pipeline import AnalysisPipeline
            output_paths: Dict[str, str] = {}
            for orientation in ["front", "back"]:
                if orientation in ori_dict:
                    image_path = str(ori_dict[orientation])
                    view_type = "Front" if orientation == "front" else "Back"
                    output_path = get_output_path(output_folder, image_path, replicate, view_type)
                    output_paths[orientation] = output_path
                    args = Namespace(
                        image=image_path,
                        input=None,
                        input_dir=None,
                        output=output_path,
                        mode="hybrid",
                        model="vit_h",
                        debug=True,
                        verbose=True,
                        advanced=True,
                        config=None,
                        orientation=orientation,
                        medium=medium,
                    )
                    pipeline = AnalysisPipeline(args)
                    pipeline.run()
            # Automatically pair front/back results if both present
            if "front" in output_paths and "back" in output_paths:
                try:
                    from colony_analysis.pairing import pair_colonies_across_views
                    pair_colonies_across_views(Path(output_paths["front"]).parent)
                except Exception as e:
                    logging.error(f"配对前后视图失败: {e}")
            if "front" not in ori_dict and "back" not in ori_dict:
                logging.warning(
                    f"{sample_name} replicate {replicate} 缺少 Front/Back 图像，跳过"
                )

            elapsed = time.time() - step_start
            logging.info(
                f"已处理 {sample_name} replicate {replicate} ({medium.upper()}) - {elapsed:.2f}s"
            )
        except Exception as e:
            logging.error(
                f"处理失败: {sample_name} replicate {replicate}, 错误: {e}"
            )
            continue

        # 解析 stats 路径
        # 需要和 get_output_path 保持一致
        # 构造 base_dir = output/Lib96_Ctrl/MMM/20250401_09191796/replicate_01
        stats_base_dir = None
        try:
            # 任选 front/back 中一个 image_path
            chosen_img = None
            for orientation in ["front", "back"]:
                if orientation in ori_dict:
                    chosen_img = str(ori_dict[orientation])
                    break
            if not chosen_img:
                continue
            filename = Path(chosen_img).stem
            match = re.match(r"(?P<group>Lib96_\w+)_@(?P<medium>\w+)_(?P<view>Back|Front)(?P<dateid>\d+)", filename)
            if not match:
                continue
            group = match.group("group")
            medium_str = match.group("medium")
            dateid = match.group("dateid")
            stats_base_dir = Path(output_folder) / group / medium_str / dateid / f"replicate_{replicate}"
        except Exception:
            continue

        front_stats = stats_base_dir / "Front" / "stats_Front.txt"
        back_stats = stats_base_dir / "Back" / "stats_Back.txt"
        combined_folder = stats_base_dir / "combined"
        combined_folder.mkdir(parents=True, exist_ok=True)
        if front_stats.exists() and back_stats.exists():
            metrics = combine_metrics(str(front_stats), str(back_stats))
            combined_path = combined_folder / "combined_stats.txt"
            with open(combined_path, "w") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")

            metrics_record = {"replicate": replicate}
            metrics_record.update(metrics)
            summary_data[(group, medium_str)].append(metrics_record)

    import csv
    import statistics

    for (group, medium), records in summary_data.items():
        if not records:
            continue
        # dateid 不唯一，summary 聚合到 group/medium/summary
        summary_dir = (
            Path(output_folder) / group / medium / "summary"
        )
        summary_dir.mkdir(parents=True, exist_ok=True)

        keys = [k for k in records[0].keys() if k != "replicate"]
        csv_path = summary_dir / "all_replicates.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["replicate"] + keys)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)

        stats_path = summary_dir / "summary_stats.txt"
        with open(stats_path, "w") as f:
            for key in keys:
                values = [rec[key] for rec in records if key in rec]
                mean = sum(values) / len(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                f.write(f"{key}: mean={mean}, std={std}\n")

    total_elapsed = time.time() - start_all
    logging.info(f"批量处理完成，总耗时 {total_elapsed:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="批量处理 R5/MMM 多角度、多重复的图像"
    )
    parser.add_argument("-i", "--input", required=True, help="原始图片根目录")
    parser.add_argument("-o", "--output", required=True, help="分析结果输出根目录")
    args = parser.parse_args()
    batch_medium_pipeline(args.input, args.output)
