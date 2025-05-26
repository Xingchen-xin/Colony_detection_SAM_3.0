# ============================================================================
# 7. colony_analysis/analysis/features.py - 特征提取器
# ============================================================================

import cv2
import numpy as np
from typing import Dict, Any, Optional


class FeatureExtractor:
    """菌落特征提取器"""

    def __init__(self, extractor_type: str = 'basic'):
        """初始化特征提取器"""
        self.extractor_type = extractor_type

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
        """提取代谢产物特征"""
        binary_mask = mask > 0

        # 提取RGB通道
        r_channel = img[:, :, 0]
        g_channel = img[:, :, 1]
        b_channel = img[:, :, 2]

        # 检测蓝色素(Actinorhodin)
        blue_mask = (b_channel > 100) & (b_channel > r_channel +
                                         20) & (b_channel > g_channel + 20) & binary_mask
        blue_area = np.sum(blue_mask)
        blue_ratio = blue_area / \
            np.sum(binary_mask) if np.sum(binary_mask) > 0 else 0

        # 检测红色素(Prodigiosin)
        red_mask = (r_channel > 100) & (r_channel > b_channel +
                                        20) & (r_channel > g_channel + 20) & binary_mask
        red_area = np.sum(red_mask)
        red_ratio = red_area / \
            np.sum(binary_mask) if np.sum(binary_mask) > 0 else 0

        # 计算色素强度
        blue_intensity = np.mean(b_channel[blue_mask]) if blue_area > 0 else 0
        red_intensity = np.mean(r_channel[red_mask]) if red_area > 0 else 0

        return {
            'metabolite_blue_area': float(blue_area),
            'metabolite_blue_ratio': float(blue_ratio),
            'metabolite_has_blue_pigment': blue_ratio > 0.05,
            'metabolite_blue_intensity_mean': float(blue_intensity),
            'metabolite_red_area': float(red_area),
            'metabolite_red_ratio': float(red_ratio),
            'metabolite_has_red_pigment': red_ratio > 0.05,
            'metabolite_red_intensity_mean': float(red_intensity)
        }
