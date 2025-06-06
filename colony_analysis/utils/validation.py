# ============================================================================
# 10. colony_analysis/utils/validation.py - 数据验证
# ============================================================================

from typing import Tuple

import cv2
import numpy as np


class ImageValidator:
    """图像验证工具"""

    @staticmethod
    def validate_image(
        img, min_size=(100, 100), max_size=(10000, 10000)
    ) -> Tuple[bool, str]:
        """验证输入图像的有效性"""
        if img is None:
            return False, "无效的图像输入(None)"

        if not isinstance(img, np.ndarray):
            return False, f"图像必须是numpy数组，而非 {type(img)}"

        if len(img.shape) < 2:
            return False, f"不支持的图像维度: {img.shape}"

        h, w = img.shape[:2]
        if h < min_size[0] or w < min_size[1]:
            return False, f"图像尺寸过小: {(h, w)}，最小要求: {min_size}"

        if h > max_size[0] or w > max_size[1]:
            return False, f"图像尺寸过大: {(h, w)}，最大允许: {max_size}"

        # 检查通道数
        if len(img.shape) == 3 and img.shape[2] not in [1, 3, 4]:
            return False, f"不支持的通道数: {img.shape[2]}"

        # 检查图像质量
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img if len(img.shape) == 2 else img[:, :, 0]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 检查图像是否过于平坦(低对比度)
        std_dev = np.std(gray)
        if std_dev < 10:
            return False, f"图像对比度过低: {std_dev:.2f}"

        return True, None


class DataValidator:
    """数据验证工具"""

    @staticmethod
    def validate_colony(colony, required_fields=None) -> Tuple[bool, str]:
        """验证菌落数据的有效性"""
        if required_fields is None:
            required_fields = ["bbox", "mask", "img"]

        for field in required_fields:
            if field not in colony:
                return False, f"缺少必需字段: {field}"

        # 检查边界框格式
        if "bbox" in colony and (
            not isinstance(colony["bbox"], tuple) or len(colony["bbox"]) != 4
        ):
            return False, f"无效的边界框格式: {colony['bbox']}"

        # 检查掩码和图像
        if "mask" in colony and "img" in colony:
            mask, img = colony["mask"], colony["img"]
            if mask.shape[:2] != img.shape[:2]:
                return False, f"掩码和图像尺寸不匹配: {mask.shape} vs {img.shape}"

        # 检查面积
        if "area" in colony and colony["area"] <= 0:
            return False, f"无效的面积值: {colony['area']}"

        return True, None
