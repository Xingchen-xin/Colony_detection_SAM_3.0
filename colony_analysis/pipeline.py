# ============================================================================
# 3. colony_analysis/pipeline.py - 分析管道
# ============================================================================

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from .analysis import ColonyAnalyzer
from .config import ConfigManager
from .core import (
    ColonyDetector,
    SAMModel,
    r5_back_analysis,
    r5_front_analysis,
    mmm_back_analysis,
    mmm_front_analysis,
    combine_metrics,
)
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

            # 6. 保存结果
            self._save_results(analyzed_colonies, img_rgb)

            # 7. 返回结果摘要
            return self._generate_summary(analyzed_colonies)

        except Exception as e:
            logging.error(f"分析管道执行失败: {e}")
            raise

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

    summary_data: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)

    for (sample_name, medium, replicate), ori_dict in groups.items():
        if "front" not in ori_dict or "back" not in ori_dict:
            logging.warning(
                f"缺少 Front 或 Back 图像: {sample_name} replicate {replicate}"
            )
            continue

        base_folder = (
            Path(output_folder)
            / sample_name
            / medium.upper()
            / f"replicate_{replicate}"
        )
        front_folder = base_folder / "Front"
        back_folder = base_folder / "Back"
        combined_folder = base_folder / "combined"
        front_folder.mkdir(parents=True, exist_ok=True)
        back_folder.mkdir(parents=True, exist_ok=True)
        combined_folder.mkdir(parents=True, exist_ok=True)

        try:
            if medium == "r5":
                r5_front_analysis(str(ori_dict["front"]), str(front_folder))
                r5_back_analysis(str(ori_dict["back"]), str(back_folder))
            elif medium == "mmm":
                mmm_front_analysis(str(ori_dict["front"]), str(front_folder))
                mmm_back_analysis(str(ori_dict["back"]), str(back_folder))
            else:
                logging.warning(f"未知培养基 '{medium}'，跳过 {sample_name}")
                continue

            logging.info(
                f"已处理 {sample_name} replicate {replicate} ({medium.upper()})"
            )
        except Exception as e:
            logging.error(
                f"处理失败: {sample_name} replicate {replicate}, 错误: {e}"
            )
            continue

        front_stats = front_folder / "stats_Front.txt"
        back_stats = back_folder / "stats_Back.txt"
        metrics = combine_metrics(str(front_stats), str(back_stats))
        combined_path = combined_folder / "combined_stats.txt"
        with open(combined_path, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        metrics_record = {"replicate": replicate}
        metrics_record.update(metrics)
        summary_data[(sample_name, medium)].append(metrics_record)

    import csv
    import statistics

    for (sample_name, medium), records in summary_data.items():
        if not records:
            continue
        summary_dir = (
            Path(output_folder) / sample_name / medium.upper() / "summary"
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

    logging.info("批量处理完成。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="批量处理 R5/MMM 多角度、多重复的图像"
    )
    parser.add_argument("-i", "--input", required=True, help="原始图片根目录")
    parser.add_argument("-o", "--output", required=True, help="分析结果输出根目录")
    args = parser.parse_args()
    batch_medium_pipeline(args.input, args.output)
