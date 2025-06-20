# ============================================================================
# 3. colony_analysis/pipeline.py - 分析管道
# ============================================================================

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from tqdm import tqdm

import cv2
import numpy as np
from colony_analysis.config.config_loader import ConfigLoader
from colony_analysis.segmenters.sam_segmenter import SamSegmenter
from colony_analysis.segmenters.unet_segmenter import UnetSegmenter
from .analysis import ColonyAnalyzer
from .config import ConfigManager
from .core import ColonyDetector, SAMModel, combine_metrics
from .utils import (
    ImageValidator,
    ResultManager,
    Visualizer,
    ImprovedVisualizer,
    collect_all_images,
    parse_filename,
)

def save_debug_images(stage: str, img: np.ndarray, masks: List[np.ndarray], debug_root: str):
    """
    在 debug_root 下按 stage 保存可视化图像。
    """
    stage_dir = os.path.join(debug_root, stage)
    os.makedirs(stage_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        vis = img.copy()
        color = [0, 255, 0] if stage.startswith("sam") else [0, 0, 255]
        vis[mask > 0] = color
        filename = os.path.join(stage_dir, f"{stage}_{i}.png")
        cv2.imwrite(filename, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def filter_sam_masks(masks: List[np.ndarray], scores: List[float], det_conf) -> List[np.ndarray]:
    """
    根据检测配置过滤SAM掩码列表，返回被保留的掩码。
    """
    filtered = []
    for mask, score in zip(masks, scores or []):
        area = int(np.sum(mask))
        if area < det_conf.min_colony_area:
            continue
        if hasattr(det_conf, "max_colony_area") and area > det_conf.max_colony_area:
            continue
        if det_conf.background_filter:
            h, w = mask.shape
            if area > h * w * det_conf.max_background_ratio:
                continue
        filtered.append(mask)
    return filtered

def validate_detection_config(config):
    """验证检测配置的数据类型"""
    try:
        # 检查并修复数值类型
        numeric_fields = [
            'min_colony_area', 'max_colony_area', 'expand_pixels',
            'adaptive_gradient_thresh', 'adaptive_expand_iters',
            'overlap_threshold', 'max_background_ratio', 'edge_contact_limit',
            'edge_margin_pixels', 'high_quality_threshold', 'supplementary_threshold',
            'shape_regularity_min', 'duplicate_centroid_threshold',
            'duplicate_overlap_threshold', 'edge_margin_ratio',
            'cross_boundary_overlap_threshold', 'min_roundness',
            'max_aspect_ratio', 'max_gray_std', 'growth_inhibited_ratio',
            'solidity_threshold'
        ]
        
        for field in numeric_fields:
            if hasattr(config, field):
                value = getattr(config, field)
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    logging.warning(f"配置字段 {field} 类型错误: {type(value)}, 值: {value}")
                    # 尝试转换
                    try:
                        if isinstance(value, dict):
                            # 如果是字典，尝试获取第一个值
                            val = list(value.values())[0] if value else 0
                            setattr(config, field, float(val))
                        else:
                            setattr(config, field, float(value))
                    except:
                        # 设置默认值
                        default_values = {
                            'min_colony_area': 800,
                            'max_colony_area': 50000,
                            'expand_pixels': 2,
                            'adaptive_gradient_thresh': 20,
                            'adaptive_expand_iters': 25,
                            'overlap_threshold': 0.4,
                            'max_background_ratio': 0.2,
                            'edge_contact_limit': 0.5,
                            'edge_margin_pixels': 20,
                            'high_quality_threshold': 0.85,
                            'supplementary_threshold': 0.5,
                            'shape_regularity_min': 0.15,
                            'duplicate_centroid_threshold': 15.0,
                            'duplicate_overlap_threshold': 0.5,
                            'edge_margin_ratio': 0.08,
                            'cross_boundary_overlap_threshold': 0.1,
                            'min_roundness': 0.3,
                            'max_aspect_ratio': 3.0,
                            'max_gray_std': 100.0,
                            'growth_inhibited_ratio': 0.30,
                            'solidity_threshold': 0.70
                        }
                        setattr(config, field, default_values.get(field, 0))
        
        # 检查布尔类型
        bool_fields = [
            'merge_overlapping', 'use_preprocessing', 'background_filter',
            'enable_edge_artifact_filter', 'enable_multi_stage',
            'enable_duplicate_merging', 'enable_adaptive_grid',
            'sort_by_quality', 'mark_cross_boundary'
        ]
        for field in bool_fields:
            if hasattr(config, field):
                value = getattr(config, field)
                if not isinstance(value, bool):
                    setattr(config, field, bool(value))
                    
    except Exception as e:
        logging.error(f"配置验证失败: {e}")





class AnalysisPipeline:
    """分析管道 - 协调整个分析流程"""

    def __init__(self, args):
        """初始化分析管道 - 修复版本"""
        # 确保device属性存在
        if not hasattr(args, "device"):
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.args = args

        # 加载并初始化配置管理器
        self.config = ConfigManager(self.args.config)
        self.config.update_from_args(self.args)

        # 初始化结果管理器
        self.result_manager = ResultManager(self.args.output)

        # 处理培养基特定的参数（安全地访问filter_medium）
        medium = getattr(self.args, 'filter_medium', None) or getattr(self.args, 'medium', 'default')
        
        # 从配置中获取培养基特定的SAM checkpoint（如果有）
        medium_params_cfg = getattr(self.config, "medium_params", {}) or {}
        sam_cfg = medium_params_cfg.get(medium.lower(), {})
        checkpoint_path = sam_cfg.get("sam", {}).get("model_path", None)
        
        # 初始化 SAM 模型
        self.sam_model = SAMModel(
            model_type=self.args.model,
            checkpoint_path=checkpoint_path,
            config=self.config,
            device=self.args.device
        )
        
        # 初始化检测器
        self.detector = ColonyDetector(
            self.sam_model,
            config=self.config,
            result_manager=self.result_manager,
            debug=self.args.debug
        )
        
        # 获取方向参数（兼容不同的参数名）
        orientation = getattr(self.args, 'side', None) or getattr(self.args, 'orientation', 'front')
        
        # 初始化分析器
        self.analyzer = ColonyAnalyzer(
            self.sam_model,
            config=self.config,
            debug=self.args.debug,
            orientation=orientation.lower() if orientation else 'front'
        )

        self.start_time = None
        
        # 配置加载器初始化
        cfg_source = self.args.config or "configs"
        if os.path.isfile(cfg_source):
            cfg_dir = str(Path(cfg_source).parent)
        else:
            cfg_dir = cfg_source
        
        # 只有在目录存在时才初始化ConfigLoader
        if os.path.exists(cfg_dir):
            try:
                self.cfg_loader = ConfigLoader(cfg_dir)
            except Exception as e:
                logging.warning(f"无法加载条件配置: {e}")
                self.cfg_loader = None
        else:
            logging.warning(f"配置目录不存在: {cfg_dir}")
            self.cfg_loader = None
            
        logging.info(f"AnalysisPipeline 初始化完成 - 培养基: {medium}, 方向: {orientation}")
        """         # Select default SAM checkpoint based on model type
        default_sam_paths = {
            "vit_b": "models/sam_vit_b_01ec64.pth",
            "vit_l": "models/sam_vit_l_0b3195.pth",
            "vit_h": "models/sam_vit_h_4b8939.pth",
        }
        model_type = getattr(self.args, "model", "vit_b")
        sam_path = self.cfg.get("model_path", default_sam_paths.get(model_type, default_sam_paths["vit_b"]))
        self.seg_sam = SamSegmenter(model_path=sam_path, model_type=self.args.model, device=self.args.device)
        unet_path = self.cfg.get('unet_model_path', "models/unet_fallback.pth")
        self.seg_unet = UnetSegmenter(model_path=unet_path, device=self.args.device) """
    def _correct_plate_perspective(self, img_rgb):
        """
        Use chessboard corner detection to attempt perspective correction
        and align the 96-well plate.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # internal corner grid size: 11x7 for a 12x8 well plate
        ret, corners = cv2.findChessboardCorners(gray, (11, 7), None)
        if ret:
            logging.info("检测到棋盘角点，进行透视校正")
            pts = corners.reshape(-1, 2)
            sums = pts.sum(axis=1)
            diffs = np.diff(pts, axis=1).ravel()
            tl = pts[np.argmin(sums)]
            br = pts[np.argmax(sums)]
            tr = pts[np.argmin(diffs)]
            bl = pts[np.argmax(diffs)]
            width = int(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))
            height = int(np.hypot(bl[0] - tl[0], bl[1] - tl[1]))
            src = np.array([tl, tr, br, bl], dtype="float32")
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img_rgb, M, (width, height))
            return warped
        else:
            logging.debug("透视校正：未检测到棋盘角点，保留原图")
            return img_rgb
    def _self_calibrate_grid(self, centroids: list[tuple], rows: int, cols: int):
        """
        Infer a 12x8 well-plate grid by clustering colony centroids into rows and columns.
        centroids: list of (x, y) tuples.
        """
        import numpy as np
        import cv2

        pts = np.array(centroids, dtype=np.float32)
        if len(pts) < rows + cols:
            logging.warning("质心数量不足，无法可靠自校准网格，使用静态网格")
            return

        # Cluster y-coordinates into row centers
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        _, labels_y, centers_y = cv2.kmeans(
            pts[:, 1].reshape(-1, 1),
            rows,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        row_centers = sorted(centers_y.ravel())

        # Cluster x-coordinates into column centers
        _, labels_x, centers_x = cv2.kmeans(
            pts[:, 0].reshape(-1, 1),
            cols,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        col_centers = sorted(centers_x.ravel())

        # Estimate radius as a fraction of minimal inter-center distance
        dy = np.min(np.diff(row_centers)) if rows > 1 else 0
        dx = np.min(np.diff(col_centers)) if cols > 1 else 0
        est_r = float(max((dx + dy) / 4, 1.0))

        # Build plate_grid mapping in dictionary format
        plate_grid = {}
        cell_h = np.min(np.diff(row_centers)) if rows > 1 else dy
        cell_w = np.min(np.diff(col_centers)) if cols > 1 else dx
        for i, y in enumerate(row_centers):
            for j, x in enumerate(col_centers):
                well_id = f"{chr(65 + i)}{j+1}"
                plate_grid[well_id] = {
                    "center": (float(y), float(x)),
                    "search_radius": est_r,
                    "row": i,
                    "col": j,
                    "expected_bbox": (
                        int(y - cell_h / 2),
                        int(x - cell_w / 2),
                        int(y + cell_h / 2),
                        int(x + cell_w / 2),
                    ),
                }

        self.config.plate_grid = plate_grid
    def run(self):
        """运行完整的分析流程"""
        self.start_time = time.time()
        try:
            # 1. 初始化组件
            self._initialize_components()
            # 加载条件化配置（medium/orientation）
            sample_path = self.args.image if self.args.image else self.args.input_dir
            cond_cfg = self.cfg_loader.load_for_image(str(sample_path))
            # —— 应用培养基特定参数覆盖 —— 
            # —— 应用条件化配置 —— 
            det_conf = getattr(self.config, "detection", None)
            if det_conf and "detection" in cond_cfg:
                for k, v in cond_cfg["detection"].items():
                    if hasattr(det_conf, k):
                        setattr(det_conf, k, v)
            if "sam" in cond_cfg:
                for k, v in cond_cfg["sam"].items():
                    if hasattr(self.seg_sam.mask_generator, k):
                        setattr(self.seg_sam.mask_generator, k, v)
            logging.info(f"Loaded condition config: {cond_cfg}")

            # 2. 加载和验证图像
            img_rgb = self._load_and_validate_image()
            # —— 全局透视校正，以对齐 96 孔板 —— 
            img_rgb = self._correct_plate_perspective(img_rgb)
            # 安全获取孔板行列数
            rows = getattr(self.args, "rows", None) or self.config.output.rows
            cols = getattr(self.args, "cols", None) or self.config.output.cols
            # —— 动态自校准 96孔网格（基于初次自动检测的质心分布） —— 
            if getattr(self.args, "force_96plate_detection", False):
                # 初次自动检测获取质心
                auto_cols = self.detector.detect(img_rgb, mode="auto")
                centroids = [c.get("centroid") for c in auto_cols if c.get("centroid") is not None]
                if centroids:
                    self._self_calibrate_grid(centroids, rows, cols)
                    logging.info("动态网格设置完成")
            else:
                if not hasattr(self.config, "plate_grid"):
                    self.config.plate_grid = self.detector._create_plate_grid(
                        img_rgb.shape[:2], rows, cols
                    )
                    logging.info(f"使用静态网格：{rows} 行 × {cols} 列")


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
                    logging.debug(f"以下孔位未检测到任何菌落：{sorted(missing)}")

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
            # 4. 如果没有检测到菌落，记录调试信息
            if not colonies:
                logging.warning("未检测到任何菌落")
                if self.args.debug:
                    self._save_debug_info(img_rgb)
                return self._generate_empty_summary()
            # 5.1 可视化检测结果
            masks = [colony["mask"] for colony in colonies]
            Visualizer.overlay_masks(
                img_rgb,
                masks,
                self.result_manager.directories["visualizations"]
            )
            # 6. 执行分析
            analyzed_colonies = self._analyze_colonies(colonies)
            
            # 7. 离群值检测
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

            # 8. 保存结果
            self._save_results(analyzed_colonies, img_rgb)

            # 9. 返回结果摘要并记录日志
            summary = self._generate_summary(analyzed_colonies)
            logging.info(
                f"分析完成: {summary['total_colonies']} 个菌落,"
                f" 耗时 {summary['elapsed_time']:.2f}s"
            )
            return summary

        except Exception as e:
            logging.error(f"分析管道执行失败: {e}")
            import traceback
            logging.debug(traceback.format_exc())

            # 返回错误摘要
            return {
                "total_colonies": 0,
                "elapsed_time": time.time() - self.start_time,
                "output_dir": self.args.output,
                "error": str(e),
                "status": "failed"
            }

    # ============================================================================
    # 修复 colony_analysis/pipeline.py 中的 _force_96plate_detection 方法
    # ============================================================================

    def _force_96plate_detection(self, img_rgb):
        """
        强制96孔板检测：修复版本
        主要修复菌落到孔位的映射逻辑
        """
        import numpy as np
        from copy import deepcopy
        import cv2
        from PIL import Image, ImageDraw, ImageFont

        # 获取plate_grid
        if not hasattr(self.config, 'plate_grid') or not self.config.plate_grid:
            self.config.plate_grid = self.detector._create_plate_grid(
                img_rgb.shape[:2], 
                getattr(self.args, "rows", 8), 
                getattr(self.args, "cols", 12)
            )
            logging.info("Created new plate grid")
        else:
            logging.info("Using existing plate grid")

        plate_grid = self.config.plate_grid
        logging.info(f"Plate grid contains {len(plate_grid)} wells")
        
        # 1) 首先执行正常的检测流程
        colonies = []
        
        # Back orientation的特殊处理
        if getattr(self.args, "orientation", "").lower() == "back":
            logging.info("Back orientation: applying color-based SAM prompts")
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            red_mask = (
                ((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170))
                & (hsv[:,:,1] > 80)
                & (hsv[:,:,2] > 80)
            )
            ys, xs = np.where(red_mask)
            pts = list(zip(xs.tolist(), ys.tolist()))
            if pts:
                step = max(1, len(pts)//16)
                for p in pts[::step]:
                    try:
                        mask, score = self.detector.segment_with_prompts(
                            img_rgb, points=[list(p)], point_labels=[1]
                        )
                        data = self.detector._extract_colony_data(
                            img_rgb, mask, f"color_{p}", "color_hint"
                        )
                        if data:
                            data["sam_score"] = float(score)
                            colonies.append(data)
                    except Exception:
                        continue
            logging.info(f"Color prompts yielded {len(colonies)} candidates")
        
        # 2) 执行混合模式检测
        if self.args.mode == "hybrid" and self.args.well_plate:
            logging.info("Hybrid mode: grid-based SAM segmentation")
            
            # 网格检测
            rows = getattr(self.args, "rows", 8)
            cols = getattr(self.args, "cols", 12)
            masks, labels = self.detector.segment_grid(
                img_rgb,
                rows=rows,
                cols=cols,
                padding=self.config.detection.edge_margin_ratio
            )
            
            # 处理网格检测结果
            for mask, lab in zip(masks, labels):
                if mask is not None and np.sum(mask) > self.config.detection.min_colony_area:
                    c = self.detector._extract_colony_data(
                        img_rgb, mask, f"grid_{lab}", "grid"
                    )
                    if c:
                        c["well_position"] = lab
                        colonies.append(c)
            
            logging.info(f"Grid segmentation added {len(colonies)} colonies")
            
            # Auto检测补充
            logging.info("Running auto-detect for additional colonies")
            auto_cols = self.detector.detect(img_rgb, mode="auto")
            logging.info(f"Auto mode found {len(auto_cols)} colonies")
            
            # 智能合并auto检测结果
            for auto_col in auto_cols:
                auto_centroid = auto_col.get('centroid', (0, 0))
                too_close = False
                
                # 检查是否与已有菌落重复
                for existing_col in colonies:
                    existing_centroid = existing_col.get('centroid', (0, 0))
                    distance = np.sqrt(
                        (auto_centroid[0] - existing_centroid[0])**2 + 
                        (auto_centroid[1] - existing_centroid[1])**2
                    )
                    if distance < 50:  # 距离阈值
                        too_close = True
                        break
                
                if not too_close:
                    colonies.append(auto_col)
        else:
            # 其他模式直接检测
            colonies.extend(self.detector.detect(img_rgb, mode=self.args.mode))
        
        logging.info(f"Total detected colonies before mapping: {len(colonies)}")
        
        # 3) **修复关键部分：将菌落分配到孔位**
        self._debug_colony_mapping(colonies, plate_grid, img_rgb)
        well_to_colony = {}  # 每个孔位只保留最佳菌落
        unmapped_colonies = []
        
        # **修复1：增加搜索半径和改进映射逻辑**
        for i, colony in enumerate(colonies):
            centroid = colony.get("centroid", None)
            if centroid is None:
                logging.warning(f"Colony {i} has no centroid, skipping")
                unmapped_colonies.append(colony)
                continue
                
            cy, cx = centroid
            min_dist = float("inf")
            min_well = None
            
            # **修复2：改进距离计算和阈值**
            for well_id, info in plate_grid.items():
                wy, wx = info["center"]
                # 使用欧几里得距离
                d = np.sqrt((cx - wx)**2 + (cy - wy)**2)
                
                # **修复3：使用更宽松的搜索半径**
                search_radius = info.get("search_radius", 50)
                max_search_radius = search_radius * 2.0  # 扩大搜索半径
                
                if d < min_dist and d < max_search_radius:
                    min_dist = d
                    min_well = well_id
            
            # **修复4：记录映射详情用于调试**
            if min_well is not None:
                logging.debug(f"Colony {i} mapped to {min_well}, distance: {min_dist:.1f}")
                
                # 如果该孔位已有菌落，比较质量
                if min_well in well_to_colony:
                    existing = well_to_colony[min_well]
                    existing_score = existing.get('sam_score', 0) * np.sqrt(existing.get('area', 0))
                    new_score = colony.get('sam_score', 0) * np.sqrt(colony.get('area', 0))
                    
                    if new_score > existing_score:
                        logging.debug(f"Replacing colony in {min_well} (better score)")
                        unmapped_colonies.append(existing)  # 旧的变成未映射
                        well_to_colony[min_well] = colony
                    else:
                        logging.debug(f"Keeping existing colony in {min_well}")
                        unmapped_colonies.append(colony)  # 新的变成未映射
                else:
                    well_to_colony[min_well] = colony
                    colony["well_position"] = min_well
            else:
                logging.debug(f"Colony {i} could not be mapped (min_dist: {min_dist:.1f})")
                unmapped_colonies.append(colony)
        
        # 4) 统计检测情况
        detected_wells = set(well_to_colony.keys())
        empty_wells = set(plate_grid.keys()) - detected_wells
        
        logging.info(f"Successfully mapped colonies to wells: {len(detected_wells)}")
        logging.info(f"Detected wells: {sorted(list(detected_wells)[:10])}...")  # 显示前10个
        logging.info(f"Empty wells: {len(empty_wells)}/96")
        logging.info(f"Unmapped colonies: {len(unmapped_colonies)}")
        
        # 5) **修复5：改进结果构建逻辑**
        forced_colonies = []
        
        # 添加成功映射的菌落
        for well_id, colony in well_to_colony.items():
            colony = deepcopy(colony)
            colony["well_position"] = well_id
            colony["forced_96plate"] = True
            colony["colony_status"] = "detected"
            colony["growth_status"] = "normal"
            
            # **确保有必要的字段**
            if "area" not in colony:
                colony["area"] = float(np.sum(colony.get("mask", np.zeros((1,1)))))
            
            forced_colonies.append(colony)
        
        # 添加未映射的菌落（如果有）- 限制数量避免过多
        for i, colony in enumerate(unmapped_colonies[:min(10, len(unmapped_colonies))]):
            colony = deepcopy(colony)
            colony["well_position"] = f"unmapped_{i}"
            colony["forced_96plate"] = True
            colony["colony_status"] = "detected_unmapped"
            forced_colonies.append(colony)
        
        # 处理空孔位
        fallback_policy = getattr(self.args, "fallback_null_policy", "fill")
        
        if fallback_policy != "skip" and len(empty_wells) > 0:
            # **修复6：改进空孔位处理**
            logging.info(f"Filling {len(empty_wells)} empty wells with inferred data")
            
            # 计算已检测菌落的平均参数
            if well_to_colony:
                areas = [colony.get("area", 0) for colony in well_to_colony.values()]
                median_area = np.median(areas) if areas else 1000
                median_radius = int(np.sqrt(median_area / np.pi)) if median_area > 0 else 25
            else:
                median_radius = 25
                median_area = np.pi * median_radius ** 2
            
            # 为空孔位生成推测条目
            for well_id in sorted(empty_wells):
                info = plate_grid[well_id]
                wy, wx = info["center"]
                wx, wy = int(wx), int(wy)
                
                # 提取局部特征
                est_radius = max(15, min(median_radius, 35))  # 限制半径范围
                mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (wx, wy), est_radius, 1, thickness=-1)
                local_pixels = img_rgb[mask > 0]
                
                if len(local_pixels) > 0:
                    local_gray = np.mean(local_pixels) if len(local_pixels.shape) > 1 else local_pixels
                    local_mean = float(np.mean(local_gray))
                else:
                    local_mean = 128.0  # 默认灰度值
                
                # 创建推测菌落
                inferred_colony = {
                    "id": f"inferred_{well_id}",
                    "well_position": well_id,
                    "centroid": (wy, wx),
                    "bbox": (wy-est_radius, wx-est_radius, wy+est_radius, wx+est_radius),
                    "area": float(np.pi * est_radius ** 2),
                    "colony_status": "inferred",
                    "growth_status": "non-growing",
                    "forced_96plate": True,
                    "is_null": True,
                    "sam_score": 0.1,  # 低分表示推测
                    "features": {
                        "area": float(np.pi * est_radius ** 2),
                        "intensity_mean": local_mean,
                    },
                    "scores": {"overall_score": 0.1},
                    "phenotype": {"development_state": "none"},
                    # 添加空的图像和掩码以避免分析时的警告
                    "img": np.zeros((est_radius*2, est_radius*2, 3), dtype=np.uint8),
                    "mask": np.zeros((est_radius*2, est_radius*2), dtype=bool),
                }
                
                forced_colonies.append(inferred_colony)
        
        logging.info(f"强制96孔板检测完成:")
        logging.info(f"  - 实际检测菌落: {len(detected_wells)} 个")
        logging.info(f"  - 推测空孔位: {len(empty_wells)} 个") 
        logging.info(f"  - 未映射菌落: {len(unmapped_colonies)} 个")
        logging.info(f"  - 总输出记录: {len(forced_colonies)} 条")
        
        # **修复7：保存调试可视化**
        if self.args.debug and len(detected_wells) > 0:
            self._save_96plate_visualization(img_rgb, well_to_colony, plate_grid, empty_wells)
        
        return forced_colonies

    def _save_96plate_visualization(self, img_rgb, well_to_colony, plate_grid, empty_wells):
        """保存96孔板检测可视化"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # 创建可视化图像
            vis_img = img_rgb.copy()
            pil_img = Image.fromarray(vis_img)
            draw = ImageDraw.Draw(pil_img)
            
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 16)
            except:
                font = None
            
            # 绘制检测到的菌落
            for well_id, colony in well_to_colony.items():
                centroid = colony.get("centroid", (0, 0))
                cy, cx = int(centroid[0]), int(centroid[1])
                
                # 绘制绿色圆圈表示检测到的菌落
                draw.ellipse([cx-20, cy-20, cx+20, cy+20], outline=(0, 255, 0), width=3)
                if font:
                    draw.text((cx-15, cy-30), well_id, fill=(0, 255, 0), font=font)
            
            # 绘制空孔位网格
            for well_id in list(empty_wells)[:20]:  # 只显示前20个空位以避免图像过乱
                info = plate_grid[well_id]
                wy, wx = info["center"]
                wx, wy = int(wx), int(wy)
                
                # 绘制红色虚线圆圈表示空孔位
                draw.ellipse([wx-15, wy-15, wx+15, wy+15], outline=(255, 0, 0), width=2)
                if font:
                    draw.text((wx-10, wy-25), well_id, fill=(255, 0, 0), font=font)
            
            # 保存可视化
            debug_dir = self.result_manager.directories.get("debug", None)
            if debug_dir:
                vis_save_path = debug_dir / "96plate_detection_mapping.png"
                pil_img.save(vis_save_path)
                logging.info(f"96孔板映射可视化已保存: {vis_save_path}")
                
        except Exception as e:
            logging.error(f"保存96孔板可视化失败: {e}")

    # ============================================================================
    # 使用说明：
    # 1. 用这个修复的方法替换 colony_analysis/pipeline.py 中的 _force_96plate_detection 方法
    # 2. 同时添加 _save_96plate_visualization 方法
    # 3. 重新运行程序
    # ============================================================================

    def _find_nearest_well(self, colony: Dict, plate_grid: Dict[str, Dict]) -> Optional[str]:
        """找到菌落最近的孔位"""
        centroid = colony.get('centroid')
        if not centroid:
            return None
        
        min_distance = float('inf')
        nearest_well = None
        
        for well_id, well_info in plate_grid.items():
            well_center = well_info['center']
            distance = np.sqrt((centroid[0] - well_center[0])**2 + 
                            (centroid[1] - well_center[1])**2)
            
            if distance < min_distance and distance < well_info.get('search_radius', 100):
                min_distance = distance
                nearest_well = well_id
        
        return nearest_well

    def _create_inferred_colony(self, well_id: str, well_info: Dict, img_rgb: np.ndarray) -> Dict:
        """为空孔位创建推测的菌落条目"""
        wy, wx = well_info["center"]
        wx, wy = int(wx), int(wy)
        
        # 提取局部区域特征
        est_radius = int(well_info.get('search_radius', 30))
        mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (wx, wy), est_radius, 1, thickness=-1)
        
        local_pixels = img_rgb[mask > 0]
        local_mean = float(np.mean(local_pixels)) if local_pixels.size > 0 else 0.0
        
        return {
            "id": f"inferred_{well_id}",
            "well_position": well_id,
            "centroid": (wy, wx),
            "area": float(np.pi * est_radius**2),
            "colony_status": "inferred",
            "is_null": True,
            "forced_96plate": True,
            "sam_score": 0.0,
            "features": {
                "area": float(np.pi * est_radius**2),
                "intensity_mean": local_mean,
            }
        }




    def _initialize_components(self):
        """初始化所有组件"""
        logging.info("初始化组件...")

        try:
            # 配置管理器
            self.config = ConfigManager(self.args.config)
            self.config.update_from_args(self.args)

            # 验证配置
            validate_detection_config(self.config.detection)

            # 应用培养基特定配置
            self._apply_medium_specific_config_safe()
        
            # SAM模型 - 添加错误处理
            try:
                self.sam_model = SAMModel(
                    model_type=self.args.model, 
                    config=self.config,
                    device=getattr(self.args, 'device', 'cuda')
                )
            except Exception as e:
                logging.error(f"SAM模型初始化失败: {e}")
                # 尝试CPU fallback
                self.sam_model = SAMModel(
                    model_type=self.args.model, 
                    config=self.config,
                    device='cpu'
                )
        
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
        
        except Exception as e:
            logging.error(f"组件初始化失败: {e}")
            raise

    def _apply_medium_specific_config_safe(self):
        """安全的培养基特定配置应用"""
        try:
            medium = getattr(self.args, "medium", "").lower()

            if medium == "r5":
                # R5 培养基适当提高最小菌落面积阈值
                current_min = getattr(self.config.detection, 'min_colony_area', 800)
                self.config.detection.min_colony_area = max(1500, current_min)

            elif medium == "mmm":
                # MMM 培养基可能菌落更小
                current_min = getattr(self.config.detection, 'min_colony_area', 800)
                self.config.detection.min_colony_area = max(1000, current_min // 2)
        
            # 安全应用外部配置
            if hasattr(self, 'cfg') and isinstance(self.cfg, dict):
                for key, value in self.cfg.items():
                    if key == 'min_colony_area' and isinstance(value, (int, float)):
                        self.config.detection.min_colony_area = int(value)
                    elif key == 'color_enhance' and isinstance(value, bool):
                        self.config.detection.use_preprocessing = value
                    
        except Exception as e:
            logging.warning(f"应用培养基配置时出错: {e}")

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
        if 'min_colony_area' in self.cfg:
            self.config.detection.min_colony_area = self.cfg['min_colony_area']
        if 'color_enhance' in self.cfg:
            self.config.detection.use_preprocessing = self.cfg['color_enhance']

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
        """执行菌落检测 - 修复版本"""
        logging.info("开始菌落检测...")

        # 验证输入图像
        if img_rgb is None or img_rgb.size == 0:
            logging.error("输入图像无效")
            return []
    
        try:
            # 在调用检测前，先验证配置
            self._validate_detection_params()
            # 直接使用现有的ColonyDetector，不要重新实现
            colonies = self.detector.detect(img_rgb, mode=self.args.mode)
            logging.info(f"检测完成，发现 {len(colonies)} 个菌落")
            
            # 验证返回的数据
            valid_colonies = []
            for i, colony in enumerate(colonies):
                if self._validate_colony_data(colony):
                    valid_colonies.append(colony)
                else:
                    logging.warning(f"菌落 {i} 数据无效，已跳过")
        
            return valid_colonies
            
        
        except Exception as e:
            logging.error(f"检测失败: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
            # 尝试使用简化的fallback方案
            try:
                logging.info("尝试fallback检测...")
                return self._detect_colonies_simple_fallback(img_rgb)
            except Exception as e2:
                logging.error(f"Fallback检测也失败: {e2}")
                return []

    def _validate_detection_params(self):
        """验证检测参数类型"""
        try:
            # 检查关键参数
            params_to_check = [
                ('min_colony_area', self.config.detection.min_colony_area),
                ('max_colony_area', self.config.detection.max_colony_area),
                ('max_background_ratio', self.config.detection.max_background_ratio),
                ('edge_contact_limit', self.config.detection.edge_contact_limit),
            ]
        
            for param_name, param_value in params_to_check:
                if isinstance(param_value, dict):
                    logging.error(f"参数 {param_name} 是字典类型: {param_value}")
                    raise TypeError(f"配置参数 {param_name} 应该是数值，但得到了字典")
                elif not isinstance(param_value, (int, float, np.integer, np.floating)):
                    logging.error(f"参数 {param_name} 类型错误: {type(param_value)}")
                    raise TypeError(f"配置参数 {param_name} 类型错误")
                
        except Exception as e:
            logging.error(f"参数验证失败: {e}")
            raise


    def _validate_colony_data(self, colony):
        """验证单个菌落数据的完整性"""
        required_fields = ['id', 'area', 'bbox', 'mask']
    
        for field in required_fields:
            if field not in colony:
                logging.warning(f"菌落数据缺少字段: {field}")
                return False
            
        # 验证数值字段
        if not isinstance(colony.get('area'), (int, float, np.integer, np.floating)):
            logging.warning(f"菌落面积类型错误: {type(colony.get('area'))}")
            return False
        
        return True

    def _detect_colonies_simple_fallback(self, img_rgb):
        """简化的fallback检测方案"""
        try:
            # 使用基本的SAM auto模式，跳过复杂的增强功能
            processed_img = self.detector._preprocess_image(img_rgb)

            # 直接使用SAM生成掩码
            masks, scores = self.detector.sam_model.segment_everything(
                processed_img, 
                min_area=500,  # 使用更宽松的阈值
                max_area=30000
            )
        
            colonies = []
            for i, (mask, score) in enumerate(zip(masks[:50], scores[:50])):  # 限制数量
                try:
                    area = np.sum(mask)
                    if 500 <= area <= 30000:  # 基本过滤
                        colony_data = self.detector._extract_colony_data(
                            img_rgb, mask, f"fallback_{i}", "fallback"
                        )
                        if colony_data:
                            colony_data["sam_score"] = float(score)
                            colonies.append(colony_data)
                except Exception:
                    continue
                
            logging.info(f"Fallback检测完成，发现 {len(colonies)} 个菌落")
            return colonies
        
        except Exception as e:
            logging.error(f"简化fallback检测失败: {e}")
            return []


    def _analyze_colonies(self, colonies):
        """执行菌落分析"""
        logging.info("开始菌落分析...")

        analyzed_colonies = self.analyzer.analyze(colonies, advanced=self.args.advanced)

        logging.info(f"分析完成，共 {len(analyzed_colonies)} 个菌落")
        return analyzed_colonies

    def _save_results(self, analyzed_colonies, img_rgb):
        """保存结果 - 增强版本，确保生成可视化"""
        logging.info("保存分析结果...")

        # 保存基本结果
        saved_files = self.result_manager.save_all_results(analyzed_colonies, self.args)
        logging.info(f"基本结果已保存: {saved_files}")

        # ===== 重要：确保生成可视化 =====
        try:
            # 导入改进的可视化模块
            from .utils.visualization import ImprovedVisualizer, save_detection_visualization
            
            # 准备样本信息
            sample_info = {
                'sample_name': getattr(self.args, 'sample_name', 'unknown'),
                'orientation': getattr(self.args, 'orientation', 'front'),
                'medium': getattr(self.args, 'medium', 'unknown')
            }
            
            # 创建可视化
            logging.info("开始生成可视化...")
            
            # 1. 使用改进的可视化函数
            save_detection_visualization(img_rgb, analyzed_colonies, 
                                    self.args.output, sample_info)
            
            # 2. 使用原有的Visualizer（如果存在）
            if hasattr(Visualizer, 'create_debug_visualizations'):
                visualizer = Visualizer(self.args.output)
                visualizer.create_debug_visualizations(img_rgb, analyzed_colonies)
            
            # 3. 额外保存一个简单的检测结果图
            self._save_simple_detection_image(img_rgb, analyzed_colonies)
            
            logging.info("可视化生成完成")
            
        except Exception as e:
            logging.error(f"生成可视化时出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            # 尝试保存最基本的图像
            try:
                self._save_basic_images(img_rgb, analyzed_colonies)
            except:
                pass

        # 生成统计报告
        self._generate_stats_file(analyzed_colonies)
        
        logging.info(f"结果已保存到: {self.args.output}")


    def _save_simple_detection_image(self, img_rgb, colonies):
        """保存简单的检测结果图像"""
        try:
            import cv2
            output_dir = Path(self.args.output)
            
            # 创建输出图像
            result_img = img_rgb.copy()
            
            # 绘制每个菌落
            for i, colony in enumerate(colonies):
                if 'bbox' in colony:
                    minr, minc, maxr, maxc = colony['bbox']
                    # 绘制边界框
                    cv2.rectangle(result_img, 
                                (int(minc), int(minr)), 
                                (int(maxc), int(maxr)), 
                                (0, 255, 0), 2)
                    
                    # 添加标签
                    label = colony.get('well_position', f'C{i}')
                    cv2.putText(result_img, str(label), 
                            (int(minc), int(minr-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (0, 255, 0), 1)
            
            # 保存图像
            output_file = output_dir / "detection_result.jpg"
            cv2.imwrite(str(output_file), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            logging.info(f"检测结果图已保存: {output_file}")
            
        except Exception as e:
            logging.error(f"保存简单检测图像失败: {e}")


    def _save_basic_images(self, img_rgb, colonies):
        """保存最基本的图像（失败安全）"""
        try:
            import cv2
            output_dir = Path(self.args.output)
            
            # 至少保存原始图像
            original_path = output_dir / "original.jpg"
            cv2.imwrite(str(original_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            
            # 保存带标记的图像
            marked = img_rgb.copy()
            for colony in colonies[:10]:  # 只标记前10个
                if 'centroid' in colony:
                    cy, cx = colony['centroid']
                    cv2.circle(marked, (int(cx), int(cy)), 20, (255, 0, 0), 3)
            
            marked_path = output_dir / "marked.jpg"
            cv2.imwrite(str(marked_path), cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
            
            logging.info(f"基本图像已保存到: {output_dir}")
            
        except Exception as e:
            logging.error(f"保存基本图像失败: {e}")


    def _generate_stats_file(self, analyzed_colonies):
        """生成统计文件"""
        try:
            stats_path = Path(self.args.output) / f"stats_{self.args.orientation}.txt"
            
            with open(stats_path, 'w') as f:
                f.write(f"Total colonies: {len(analyzed_colonies)}\n")
                
                if analyzed_colonies:
                    areas = [c.get('area', 0) for c in analyzed_colonies]
                    f.write(f"Average area: {np.mean(areas):.2f}\n")
                    f.write(f"Area range: {min(areas):.2f} - {max(areas):.2f}\n")
                    
                    # 统计每个孔位的菌落
                    well_counts = {}
                    for c in analyzed_colonies:
                        well = c.get('well_position', 'unmapped')
                        well_counts[well] = well_counts.get(well, 0) + 1
                    
                    f.write(f"\nColonies per well:\n")
                    for well, count in sorted(well_counts.items()):
                        f.write(f"  {well}: {count}\n")
            
            logging.info(f"统计文件已保存: {stats_path}")
            
        except Exception as e:
            logging.error(f"生成统计文件失败: {e}")

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


    def _generate_empty_summary(self):
        """生成空结果摘要"""
        return {
            "total_colonies": 0,
            "elapsed_time": time.time() - self.start_time,
            "output_dir": self.args.output,
            "mode": self.args.mode,
            "model": self.args.model,
            "advanced": self.args.advanced,
            "status": "no_colonies_detected"
        }

    def _save_debug_info(self, img_rgb):
        """保存调试信息"""
        try:
            debug_dir = self.result_manager.directories.get("debug", "debug")
            os.makedirs(debug_dir, exist_ok=True)
        
            # 保存原始图像
            import cv2
            cv2.imwrite(
                os.path.join(debug_dir, "original_image.jpg"),
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            )
        
            # 保存配置信息
            config_info = {
                "mode": self.args.mode,
                "model": self.args.model,
                "min_colony_area": getattr(self.config.detection, 'min_colony_area', 'unknown'),
                "device": str(self.sam_model.device) if self.sam_model else 'unknown'
            }
        
            with open(os.path.join(debug_dir, "config_info.json"), 'w') as f:
                json.dump(config_info, f, indent=2)
            
        except Exception as e:
            logging.error(f"保存调试信息失败: {e}")
    def _debug_colony_mapping(self, colonies, plate_grid, img_rgb):
        """
        调试菌落映射问题的专用方法
        """
        import cv2
        import numpy as np
        
        logging.info(f"=== 开始菌落映射调试 ===")
        logging.info(f"检测到的菌落数量: {len(colonies)}")
        logging.info(f"网格孔位数量: {len(plate_grid)}")
        
        # 1. 检查图像和网格的坐标范围
        height, width = img_rgb.shape[:2]
        logging.info(f"图像尺寸: {height} x {width}")
        
        # 2. 检查网格的坐标范围
        if plate_grid:
            centers_y = [info["center"][0] for info in plate_grid.values()]
            centers_x = [info["center"][1] for info in plate_grid.values()]
            
            logging.info(f"网格Y坐标范围: {min(centers_y):.1f} - {max(centers_y):.1f}")
            logging.info(f"网格X坐标范围: {min(centers_x):.1f} - {max(centers_x):.1f}")
            
            # 检查几个关键孔位
            key_wells = ["A1", "A12", "H1", "H12", "D6"]
            for well_id in key_wells:
                if well_id in plate_grid:
                    info = plate_grid[well_id]
                    cy, cx = info["center"]
                    radius = info["search_radius"]
                    logging.info(f"孔位 {well_id}: 中心=({cy:.1f}, {cx:.1f}), 搜索半径={radius:.1f}")
        
        # 3. 检查菌落的坐标范围
        if colonies:
            colony_centroids = [c.get("centroid", (0, 0)) for c in colonies if c.get("centroid")]
            if colony_centroids:
                colony_y = [c[0] for c in colony_centroids]
                colony_x = [c[1] for c in colony_centroids]
                
                logging.info(f"菌落Y坐标范围: {min(colony_y):.1f} - {max(colony_y):.1f}")
                logging.info(f"菌落X坐标范围: {min(colony_x):.1f} - {max(colony_x):.1f}")
                
                # 显示前5个菌落的坐标
                for i, colony in enumerate(colonies[:5]):
                    centroid = colony.get("centroid", (0, 0))
                    area = colony.get("area", 0)
                    logging.info(f"菌落 {i}: 中心=({centroid[0]:.1f}, {centroid[1]:.1f}), 面积={area:.0f}")
        
        # 4. 测试映射逻辑
        if colonies and plate_grid:
            logging.info("=== 测试映射逻辑 ===")
            
            test_colony = colonies[0]
            centroid = test_colony.get("centroid", (0, 0))
            cy, cx = centroid
            
            distances = []
            for well_id, info in plate_grid.items():
                wy, wx = info["center"]
                distance = np.sqrt((cx - wx)**2 + (cy - wy)**2)
                search_radius = info.get("search_radius", 50)
                distances.append((well_id, distance, search_radius, distance < search_radius))
            
            distances.sort(key=lambda x: x[1])
            
            logging.info(f"测试菌落坐标: ({cy:.1f}, {cx:.1f})")
            logging.info("最近的10个孔位:")
            for well_id, dist, radius, within in distances[:10]:
                status = "✓" if within else "✗"
                logging.info(f"  {status} {well_id}: 距离={dist:.1f}, 搜索半径={radius:.1f}")
        
        # 5. 保存调试可视化
        if self.args.debug:
            self._save_mapping_debug_visualization(colonies, plate_grid, img_rgb)
        
        logging.info(f"=== 菌落映射调试结束 ===")

    def _save_mapping_debug_visualization(self, colonies, plate_grid, img_rgb):
        """保存映射调试可视化"""
        try:
            import cv2
            from PIL import Image, ImageDraw, ImageFont
            
            # 创建调试图像
            debug_img = img_rgb.copy()
            pil_img = Image.fromarray(debug_img)
            draw = ImageDraw.Draw(pil_img)
            
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 12)
            except:
                font = None
            
            # 绘制网格中心点（蓝色）
            for well_id, info in plate_grid.items():
                wy, wx = info["center"]
                radius = info.get("search_radius", 50)
                
                # 绘制搜索范围（淡蓝色圆圈）
                draw.ellipse([wx-radius, wy-radius, wx+radius, wy+radius], 
                            outline=(100, 150, 255), width=1)
                
                # 绘制中心点（蓝色小点）
                draw.ellipse([wx-3, wy-3, wx+3, wy+3], fill=(0, 0, 255))
                
                # 标注孔位ID
                if font and well_id in ["A1", "A12", "H1", "H12", "D6"]:  # 只标注关键孔位
                    draw.text((wx+5, wy-15), well_id, fill=(0, 0, 255), font=font)
            
            # 绘制检测到的菌落（红色）
            for i, colony in enumerate(colonies):
                centroid = colony.get("centroid", (0, 0))
                cy, cx = int(centroid[0]), int(centroid[1])
                area = colony.get("area", 0)
                
                # 根据面积大小绘制不同大小的圆圈
                size = max(5, min(25, int(np.sqrt(area) / 10)))
                
                # 绘制菌落位置（红色圆圈）
                draw.ellipse([cx-size, cy-size, cx+size, cy+size], 
                            outline=(255, 0, 0), width=2)
                
                # 标注菌落编号（前10个）
                if i < 10 and font:
                    draw.text((cx+size+2, cy-size), str(i), fill=(255, 0, 0), font=font)
            
            # 保存调试图像
            debug_dir = self.result_manager.directories.get("debug", None)
            if debug_dir:
                debug_path = debug_dir / "colony_mapping_debug.png"
                pil_img.save(debug_path)
                logging.info(f"映射调试可视化已保存: {debug_path}")
                
                # 同时保存一个标注版本
                annotated_path = debug_dir / "colony_mapping_annotated.png"
                
                # 在图像上添加图例
                legend_y = 30
                draw.text((10, legend_y), "蓝色圆圈: 网格孔位搜索范围", fill=(0, 0, 255), font=font)
                draw.text((10, legend_y+20), "蓝色点: 孔位中心", fill=(0, 0, 255), font=font)
                draw.text((10, legend_y+40), "红色圆圈: 检测到的菌落", fill=(255, 0, 0), font=font)
                
                pil_img.save(annotated_path)
                logging.info(f"带注释的调试图像已保存: {annotated_path}")
                
        except Exception as e:
            logging.error(f"保存映射调试可视化失败: {e}")






def batch_medium_pipeline(img_paths: List[Path], output_folder: str, **kwargs):
    """批量处理指定的图像列表 - 修复版本"""
    if not img_paths:
        logging.warning("没有图像需要处理")
        return

    from collections import defaultdict
    groups: Dict[Tuple[str, str, str], Dict[str, Path]] = defaultdict(dict)

    # 组织图像
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
    
    for (sample_name, medium, replicate), ori_dict in tqdm(
        groups.items(), desc="Batch processing", ncols=80
    ):
        step_start = time.time()

        try:
            # --- Use actual AnalysisPipeline for each orientation ---
            from argparse import Namespace
            
            output_paths: Dict[str, str] = {}
            
            for orientation in ["front", "back"]:
                if orientation in ori_dict:
                    image_path = str(ori_dict[orientation])
                    view_type = "Front" if orientation == "front" else "Back"
                    output_path = get_output_path(output_folder, image_path, replicate, view_type)
                    output_paths[orientation] = output_path
                    
                    # 创建完整的参数对象，包括所有必要的属性
                    args = Namespace(
                        # 基本参数
                        image=image_path,
                        input=None,
                        input_dir=None,
                        output=output_path,
                        
                        # 检测参数
                        mode=kwargs.get('mode', 'hybrid'),
                        model=kwargs.get('model', 'vit_b'),
                        device=kwargs.get('device', 'cuda'),
                        min_area=kwargs.get('min_area', 800),
                        
                        # 培养基和方向参数（重要！）
                        medium=medium.lower(),  # 确保小写
                        orientation=orientation,
                        side=orientation,  # 有些地方使用side而不是orientation
                        filter_medium=medium.lower(),  # 修复：添加缺失的属性
                        
                        # 96孔板参数
                        well_plate=kwargs.get('well_plate', True),
                        rows=kwargs.get('rows', 8),
                        cols=kwargs.get('cols', 12),
                        force_96plate_detection=kwargs.get('force_96plate_detection', False),
                        fallback_null_policy=kwargs.get('fallback_null_policy', 'fill'),
                        
                        # 分析参数
                        advanced=kwargs.get('advanced', False),
                        debug=kwargs.get('debug', False),
                        verbose=kwargs.get('verbose', False),
                        
                        # 其他参数
                        config=kwargs.get('config', None),
                        sample_name=sample_name,
                        replicate=replicate,
                        
                        # 离群值检测参数
                        outlier_detection=kwargs.get('outlier_detection', False),
                        outlier_metric=kwargs.get('outlier_metric', 'area'),
                        outlier_threshold=kwargs.get('outlier_threshold', 3.0),
                    )
                    
                    # 运行分析管道
                    logging.info(f"处理 {sample_name} {medium} {orientation} replicate {replicate}")
                    pipeline = AnalysisPipeline(args)
                    results = pipeline.run()
                    
                    # 检查结果
                    if results and results.get('total_colonies', 0) > 0:
                        logging.info(f"成功检测到 {results['total_colonies']} 个菌落")
                    else:
                        logging.warning(f"未检测到菌落: {sample_name} {medium} {orientation}")
                        
            # 自动配对前后视图结果（如果两者都存在）
            if "front" in output_paths and "back" in output_paths:
                try:
                    from colony_analysis.pairing import pair_colonies_across_views
                    # 配对应该在replicate级别进行
                    replicate_dir = Path(output_paths["front"]).parent
                    pair_colonies_across_views(str(replicate_dir))
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
                f"处理失败: {sample_name} {medium} replicate {replicate}, 错误: {e}"
            )
            import traceback
            logging.error(traceback.format_exc())
            continue

        # 生成统计文件
        try:
            stats_base_dir = None
            # 构造统计文件路径
            chosen_img = None
            for orientation in ["front", "back"]:
                if orientation in ori_dict:
                    chosen_img = str(ori_dict[orientation])
                    break
                    
            if chosen_img:
                filename = Path(chosen_img).stem
                match = re.match(r"(?P<group>Lib96_\w+)_@(?P<medium>\w+)_(?P<view>Back|Front)(?P<dateid>\d+)", filename)
                if match:
                    group = match.group("group")
                    medium_str = match.group("medium")
                    dateid = match.group("dateid")
                    stats_base_dir = Path(output_folder) / group / medium_str / dateid / f"replicate_{replicate}"
                    
                    # 合并前后统计
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
                        
        except Exception as e:
            logging.error(f"生成统计文件失败: {e}")

    # 生成汇总报告
    import csv
    import statistics

    for (group, medium), records in summary_data.items():
        if not records:
            continue
            
        summary_dir = Path(output_folder) / group / medium / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        keys = [k for k in records[0].keys() if k != "replicate"]
        csv_path = summary_dir / "all_replicates.csv"
        
        try:
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["replicate"] + keys)
                writer.writeheader()
                for rec in records:
                    writer.writerow(rec)

            stats_path = summary_dir / "summary_stats.txt"
            with open(stats_path, "w") as f:
                for key in keys:
                    values = [rec[key] for rec in records if key in rec]
                    if values:
                        mean = sum(values) / len(values)
                        std = statistics.stdev(values) if len(values) > 1 else 0.0
                        f.write(f"{key}: mean={mean}, std={std}\n")
        except Exception as e:
            logging.error(f"生成汇总报告失败: {e}")

    total_elapsed = time.time() - start_all
    logging.info(f"批量处理完成，总耗时 {total_elapsed:.2f}s")



# ============================================================================
# 添加到 colony_analysis/pipeline.py 中的调试方法
# ============================================================================





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="批量处理 R5/MMM 多角度、多重复的图像"
    )
    parser.add_argument("-i", "--input", required=True, help="原始图片根目录")
    parser.add_argument("-o", "--output", required=True, help="分析结果输出根目录")
    args = parser.parse_args()
    batch_medium_pipeline(args.input, args.output)
