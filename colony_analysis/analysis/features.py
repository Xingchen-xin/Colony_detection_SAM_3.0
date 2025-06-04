# ============================================================================
# 7. colony_analysis/analysis/features.py - 特征提取器
# ============================================================================

import cv2
import numpy as np
import os
from typing import Dict, Any, Optional


class FeatureExtractor:
    """菌落特征提取器"""

    def __init__(self, extractor_type: str = 'basic', debug: bool = False):
        """初始化特征提取器"""
        self.extractor_type = extractor_type
        self.debug = debug
        # 如果启用了调试功能，创建用于保存调试图像的目录
        if self.debug:
            self.debug_dir = "debug_metabolite"
            os.makedirs(self.debug_dir, exist_ok=True)

    def extract(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """提取特征"""
        if self.extractor_type == 'basic':
            return self._extract_basic_features(img, mask)
        elif self.extractor_type == 'aerial':
            return self._extract_aerial_features(img, mask)
        elif self.extractor_type == 'metabolite':
            return self._extract_metabolite_features(img, mask)
        else:
            return {}

    def _extract_basic_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """提取基本形态特征"""
        binary_mask = mask > 0
        area = np.sum(binary_mask)

        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        features = {'area': float(area)}

        if contours:
            contour = max(contours, key=cv2.contourArea)

            # 周长
            perimeter = cv2.arcLength(contour, True)
            features['perimeter'] = float(perimeter)

            # 圆形度
            if perimeter > 0:
                circularity = (4 * np.pi * cv2.contourArea(contour)
                               ) / (perimeter * perimeter)
                features['circularity'] = float(circularity)

            # 长宽比
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                features['aspect_ratio'] = float(aspect_ratio)

            # 凸性
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convexity = cv2.contourArea(contour) / hull_area
                features['convexity'] = float(convexity)

        # 边缘密度
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(
            img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges[binary_mask]) / area if area > 0 else 0
        features['edge_density'] = float(edge_density)

        return features

    def _extract_aerial_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """提取气生菌丝特征"""
        binary_mask = mask > 0

        # 转换到HSV空间
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        _, s, v = cv2.split(hsv)

        # 识别气生菌丝
        aerial_mask = (v > 200) & (s < 50) & binary_mask
        aerial_ratio = np.sum(
            aerial_mask) / np.sum(binary_mask) if np.sum(binary_mask) > 0 else 0

        # 计算"高度"
        if np.sum(aerial_mask) > 0:
            aerial_height_mean = np.mean(v[aerial_mask])
            aerial_height_std = np.std(v[aerial_mask])
            aerial_height_max = np.max(v[aerial_mask])
        else:
            aerial_height_mean = aerial_height_std = aerial_height_max = 0

        return {
            'morphology_aerial_area': float(np.sum(aerial_mask)),
            'morphology_aerial_ratio': float(aerial_ratio),
            'morphology_aerial_height_mean': float(aerial_height_mean),
            'morphology_aerial_height_std': float(aerial_height_std),
            'morphology_aerial_height_max': float(aerial_height_max)
        }

    def _extract_metabolite_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        提取代谢产物特征：
        - 先对培养基进行CLAHE均衡化减弱背景黄褐色
        - 在菌落边缘检测蓝色色素（actinorhodin），
        - 在菌落内部检测红色色素（prodigiosin），
        - 输出面积、比例与平均强度指标
        """
        binary_mask = mask > 0
        total_pixels = np.sum(binary_mask)
        features: Dict[str, Any] = {
            'metabolite_blue_area': 0.0,
            'metabolite_blue_ratio': 0.0,
            'metabolite_blue_intensity_mean': 0.0,
            'metabolite_red_area': 0.0,
            'metabolite_red_ratio': 0.0,
            'metabolite_red_intensity_mean': 0.0,
        }

        if total_pixels == 0:
            return features

        # 1) 对图像进行CLAHE均衡化，减弱培养基干扰
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = cv2.split(img_hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_ch_eq = clahe.apply(v_ch)
        hsv_eq = cv2.merge([h_ch, s_ch, v_ch_eq])
        img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

        # 2) 提取边缘区域用于蓝色检测
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=2)
        edge_mask = binary_mask.astype(np.uint8) - eroded
        edge_mask_bool = edge_mask > 0

        # 3) 在边缘区域检测蓝色色素 (actinorhodin)
        r_channel = img_eq[:, :, 0].astype(np.int32)
        g_channel = img_eq[:, :, 1].astype(np.int32)
        b_channel = img_eq[:, :, 2].astype(np.int32)
        hsv_eq_full = cv2.cvtColor(img_eq, cv2.COLOR_RGB2HSV)
        h_full, s_full, v_full_eq = cv2.split(hsv_eq_full)

        blue_threshold = (
            (b_channel > r_channel + 20) &
            (b_channel > g_channel + 20) &
            (h_full >= 90) & (h_full <= 140) &
            edge_mask_bool
        )

        # 调试：将蓝色阈值区域生成可视化图，并保存
        if self.debug:
            # 生成叠加图：在 img_eq 上标记蓝区为红色
            vis_blue = img_eq.copy()
            vis_blue[blue_threshold] = [255, 0, 0]  # 用红色高亮蓝色区域
            # 计算质心用于命名
            ys_b, xs_b = np.where(blue_threshold)
            if len(ys_b) > 0:
                cy_b = int(np.mean(ys_b))
                cx_b = int(np.mean(xs_b))
            else:
                cy_b, cx_b = 0, 0
            filename_blue = f"blue_{cy_b}_{cx_b}.png"
            cv2.imwrite(os.path.join(self.debug_dir, filename_blue),
                        cv2.cvtColor(vis_blue, cv2.COLOR_RGB2BGR))

        blue_area = np.sum(blue_threshold)
        features['metabolite_blue_area'] = float(blue_area)
        features['metabolite_blue_ratio'] = float(blue_area / total_pixels)
        features['metabolite_blue_intensity_mean'] = float(
            np.mean(b_channel[blue_threshold]) if blue_area > 0 else 0.0
        )

        # 4) 在内部区域检测红色色素 (prodigiosin)
        interior_mask_bool = binary_mask & (~edge_mask_bool)

        red_threshold = (
            (r_channel > b_channel + 20) &
            (r_channel > g_channel + 20) &
            (
                ((h_full >= 0) & (h_full <= 10)) |
                ((h_full >= 170) & (h_full <= 179))
            ) &
            (s_full > 80) &
            (v_full_eq > 80) &
            interior_mask_bool
        )

        # 调试：将红色阈值区域生成可视化图，并保存
        if self.debug:
            vis_red = img_eq.copy()
            vis_red[red_threshold] = [0, 0, 255]  # 用蓝色高亮红色区域
            ys_r, xs_r = np.where(red_threshold)
            if len(ys_r) > 0:
                cy_r = int(np.mean(ys_r))
                cx_r = int(np.mean(xs_r))
            else:
                cy_r, cx_r = 0, 0
            filename_red = f"red_{cy_r}_{cx_r}.png"
            cv2.imwrite(os.path.join(self.debug_dir, filename_red),
                        cv2.cvtColor(vis_red, cv2.COLOR_RGB2BGR))

        red_area = np.sum(red_threshold)
        features['metabolite_red_area'] = float(red_area)
        features['metabolite_red_ratio'] = float(red_area / total_pixels)
        features['metabolite_red_intensity_mean'] = float(
            np.mean(r_channel[red_threshold]) if red_area > 0 else 0.0
        )

        return features
