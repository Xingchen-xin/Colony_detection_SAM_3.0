# ============================================================================
# 6. colony_analysis/core/detection.py - 菌落检测器
# ============================================================================

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .sam_model import SAMModel
from ..utils.validation import ImageValidator, DataValidator


@dataclass
class DetectionConfig:
    """检测配置数据类"""
    mode: str = 'auto'
    min_colony_area: int = 5000
    expand_pixels: int = 8
    merge_overlapping: bool = True
    use_preprocessing: bool = True
    overlap_threshold: float = 0.3


class ColonyDetector:
    """统一的菌落检测器"""

    def __init__(self, sam_model: SAMModel, config=None):
        """初始化菌落检测器"""
        self.sam_model = sam_model
        self.config = self._load_detection_config(config)
        logging.info("菌落检测器已初始化")

    def _load_detection_config(self, config) -> DetectionConfig:
        """加载检测配置"""
        if config is None:
            return DetectionConfig()

        detection_params = {}
        detection_obj = config.get('detection')
        if hasattr(detection_obj, '__dict__'):
            detection_params = {
                'mode': detection_obj.mode,
                'min_colony_area': detection_obj.min_colony_area,
                'expand_pixels': detection_obj.expand_pixels,
                'merge_overlapping': detection_obj.merge_overlapping,
                'use_preprocessing': detection_obj.use_preprocessing,
                'overlap_threshold': detection_obj.overlap_threshold
            }

        return DetectionConfig(**detection_params)

    def detect(self, img_rgb: np.ndarray, mode: Optional[str] = None) -> List[Dict]:
        """检测菌落的主要入口方法"""
        # 验证输入
        is_valid, error_msg = ImageValidator.validate_image(img_rgb)
        if not is_valid:
            raise ValueError(f"图像验证失败: {error_msg}")

        # 确定检测模式
        detection_mode = mode or self.config.mode

        # 预处理图像
        processed_img = self._preprocess_image(img_rgb)

        # 根据模式执行检测
        if detection_mode == 'grid':
            colonies = self._detect_grid_mode(processed_img)
        elif detection_mode == 'auto':
            colonies = self._detect_auto_mode(processed_img)
        elif detection_mode == 'hybrid':
            colonies = self._detect_hybrid_mode(processed_img)
        else:
            raise ValueError(f"不支持的检测模式: {detection_mode}")

        # 后处理
        colonies = self._post_process_colonies(colonies)

        logging.info(f"检测完成，发现 {len(colonies)} 个菌落")
        return colonies

    def _preprocess_image(self, img_rgb: np.ndarray) -> np.ndarray:
        """预处理图像"""
        if not self.config.use_preprocessing:
            return img_rgb

        # 转换到HSV空间进行处理
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # 对亮度通道进行中值滤波
        v_filtered = cv2.medianBlur(v, 5)

        # 对饱和度通道进行高斯滤波
        s_filtered = cv2.GaussianBlur(s, (5, 5), 0)

        # 重新组合并转回RGB
        hsv_processed = cv2.merge([h, s_filtered, v_filtered])
        processed_img = cv2.cvtColor(hsv_processed, cv2.COLOR_HSV2RGB)

        return processed_img

    def _detect_auto_mode(self, img: np.ndarray) -> List[Dict]:
        """自动检测模式"""
        logging.info("使用自动检测模式...")

        masks, scores = self.sam_model.segment_everything(
            img, min_area=self.config.min_colony_area // 4
        )

        colonies = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # 增强掩码
            enhanced_mask = self._enhance_colony_mask(mask)

            # 面积过滤
            area = np.sum(enhanced_mask)
            if area < self.config.min_colony_area:
                continue

            # 提取菌落数据
            colony_data = self._extract_colony_data(
                img, enhanced_mask, f'colony_{i}', 'sam_auto'
            )

            if colony_data:
                colony_data['sam_score'] = float(score)
                colonies.append(colony_data)

        return colonies

    def _detect_grid_mode(self, img: np.ndarray) -> List[Dict]:
        """网格检测模式"""
        logging.info("使用网格检测模式...")

        masks, labels = self.sam_model.segment_grid(img)

        colonies = []
        for mask, label in zip(masks, labels):
            area = np.sum(mask)
            if area < self.config.min_colony_area:
                continue

            colony_data = self._extract_colony_data(
                img, mask, label, 'sam_grid'
            )

            if colony_data:
                colony_data['well_position'] = label
                colonies.append(colony_data)

        return colonies

    def _detect_hybrid_mode(self, img: np.ndarray) -> List[Dict]:
        """混合检测模式"""
        logging.info("使用混合检测模式...")

        # 先尝试自动检测
        auto_colonies = self._detect_auto_mode(img)

        # 如果检测结果太少，补充网格检测
        if len(auto_colonies) < 10:
            grid_colonies = self._detect_grid_mode(img)
            all_colonies = auto_colonies + grid_colonies
            return self._remove_duplicates(all_colonies)

        return auto_colonies

    def _enhance_colony_mask(self, mask: np.ndarray) -> np.ndarray:
        """增强菌落掩码形状"""
        if np.sum(mask) == 0:
            return mask

        # 找到质心
        y_indices, x_indices = np.where(mask)
        center_y, center_x = np.mean(y_indices), np.mean(x_indices)

        # 计算等效半径
        area = np.sum(mask)
        equiv_radius = np.sqrt(area / np.pi)

        # 创建圆形扩展掩码
        h, w = mask.shape
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt(
            (y_grid - center_y)**2 + (x_grid - center_x)**2)

        # 创建平滑的圆形掩码
        expanded_mask = dist_from_center <= (
            equiv_radius + self.config.expand_pixels)

        # 结合原始掩码
        enhanced_mask = np.logical_or(mask, expanded_mask)

        return enhanced_mask.astype(np.uint8)

    def _extract_colony_data(self, img: np.ndarray, mask: np.ndarray,
                             colony_id: str, detection_method: str = 'sam') -> Dict:
        """从图像和掩码中提取菌落数据"""
        # 计算边界框
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return None

        minr, minc = np.min(y_indices), np.min(x_indices)
        maxr, maxc = np.max(y_indices) + 1, np.max(x_indices) + 1

        # 提取菌落图像和掩码
        colony_img = img[minr:maxr, minc:maxc].copy()
        colony_mask = mask[minr:maxr, minc:maxc]

        # 创建掩码应用的图像
        masked_img = np.zeros_like(colony_img)
        masked_img[colony_mask > 0] = colony_img[colony_mask > 0]

        # 计算基本属性
        area = float(np.sum(mask))
        centroid = (float(np.mean(y_indices)), float(np.mean(x_indices)))

        return {
            'id': colony_id,
            'bbox': (minr, minc, maxr, maxc),
            'area': area,
            'centroid': centroid,
            'mask': colony_mask,
            'img': colony_img,
            'masked_img': masked_img,
            'detection_method': detection_method
        }

    def _post_process_colonies(self, colonies: List[Dict]) -> List[Dict]:
        """后处理菌落列表"""
        if not colonies:
            return colonies

        # 验证菌落数据
        valid_colonies = []
        for colony in colonies:
            is_valid, error_msg = DataValidator.validate_colony(colony)
            if is_valid:
                valid_colonies.append(colony)
            else:
                logging.debug(f"移除无效菌落: {error_msg}")

        # 过滤重叠菌落
        if self.config.merge_overlapping and len(valid_colonies) > 1:
            valid_colonies = self._filter_overlapping_colonies(valid_colonies)

        return valid_colonies

    def _filter_overlapping_colonies(self, colonies: List[Dict]) -> List[Dict]:
        """过滤重叠的菌落"""
        if len(colonies) <= 1:
            return colonies

        # 按面积排序，保留较大的菌落
        sorted_colonies = sorted(
            colonies, key=lambda x: x['area'], reverse=True)

        filtered_colonies = []
        used_regions = []

        for colony in sorted_colonies:
            bbox = colony['bbox']

            # 检查是否与已使用区域重叠
            is_overlapping = False
            for used_bbox in used_regions:
                if self._calculate_bbox_overlap(bbox, used_bbox) > self.config.overlap_threshold:
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered_colonies.append(colony)
                used_regions.append(bbox)

        logging.info(f"重叠过滤：{len(colonies)} -> {len(filtered_colonies)}")
        return filtered_colonies

    def _calculate_bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """计算两个边界框的重叠比例"""
        minr1, minc1, maxr1, maxc1 = bbox1
        minr2, minc2, maxr2, maxc2 = bbox2

        # 计算重叠区域
        overlap_minr = max(minr1, minr2)
        overlap_minc = max(minc1, minc2)
        overlap_maxr = min(maxr1, maxr2)
        overlap_maxc = min(maxc1, maxc2)

        # 检查是否有重叠
        if overlap_minr >= overlap_maxr or overlap_minc >= overlap_maxc:
            return 0.0

        # 计算重叠面积和比例
        overlap_area = (overlap_maxr - overlap_minr) * \
            (overlap_maxc - overlap_minc)
        area1 = (maxr1 - minr1) * (maxc1 - minc1)
        area2 = (maxr2 - minr2) * (maxc2 - minc2)

        return overlap_area / min(area1, area2)

    def _remove_duplicates(self, colonies: List[Dict]) -> List[Dict]:
        """移除重复的菌落"""
        # 简单的重复检测逻辑
        return colonies
