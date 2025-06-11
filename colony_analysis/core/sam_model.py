# ============================================================================
# 5. colony_analysis/core/sam_model.py - SAM模型封装
# ============================================================================
import inspect
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from tqdm import tqdm



class SAMModel:
    """统一的SAM模型封装类"""

    def __init__(self, model_type="vit_b", checkpoint_path=None, config=None, device: Optional[str] = None):
        """初始化SAM模型"""
        self.model_type = model_type
        # Allow override via constructor, fallback to CUDA if available
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested)
        self.checkpoint_path = self._resolve_checkpoint_path(
            checkpoint_path, model_type
        )
        self.params = self._extract_sam_params(config)

        self._load_model()
        logging.info(f"SAM模型已初始化 ({model_type})，设备: {self.device}")

    def _load_model(self):
        """
        加载SAM模型到指定设备，手动加载权重以确保 map_location 正确。
        """
        # Instantiate SAM model class without loading checkpoint
        ModelClass = sam_model_registry[self.model_type]
        sig = inspect.signature(ModelClass)
        if 'checkpoint' in sig.parameters:
            self.sam = ModelClass(checkpoint=None)
        else:
            self.sam = ModelClass()
        # Manually load the checkpoint with map_location
        state = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        self.sam.load_state_dict(state, strict=False)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, **self.params)

    def _resolve_checkpoint_path(
        self, checkpoint_path: Optional[str], model_type: str
    ) -> str:
        """解析检查点路径"""
        if checkpoint_path and Path(checkpoint_path).exists():
            return checkpoint_path

        # 默认路径映射
        default_paths = {
            "vit_h": "models/sam_vit_h_4b8939.pth",
            "vit_l": "models/sam_vit_l_0b3195.pth",
            "vit_b": "models/sam_vit_b_01ec64.pth",
        }

        path = default_paths.get(model_type)
        possible_paths = [
            path,
            f"src/{path}",
            Path.home() / ".colony_analysis" / "models" / Path(path).name,
        ]

        for p in possible_paths:
            if Path(p).exists():
                return str(p)

        raise FileNotFoundError(f"找不到模型文件。请下载 {model_type} 模型到: {path}")

    def _extract_sam_params(self, config) -> dict:
        """从配置中提取SAM参数"""
        default_params = {
            "points_per_side": 64,
            "pred_iou_thresh": 0.85,
            "stability_score_thresh": 0.8,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 1500,
        }

        if config is not None:
            try:
                sam_config = config.get("sam")
                if hasattr(sam_config, "__dict__"):
                    # 如果是dataclass对象
                    for key in default_params:
                        if hasattr(sam_config, key):
                            default_params[key] = getattr(sam_config, key)
                elif isinstance(sam_config, dict):
                    default_params.update(sam_config)
            except Exception as e:
                logging.warning(f"获取SAM配置参数失败: {e}")

        return default_params


    def segment_everything(
        self, image: np.ndarray, min_area: int = 25, max_area: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """自动分割图像中的所有区域"""
        # 预处理图像
        image = self._preprocess_image(image)

        # 生成掩码
        masks_data = self.mask_generator.generate(image)

        # 过滤和提取结果
        masks = []
        scores = []

        for mask_data in tqdm(masks_data, desc="SAM 分割候选", ncols=80):
            mask = mask_data["segmentation"]
            score = mask_data["stability_score"]
            area = mask_data["area"]

            # 面积过滤
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            masks.append(mask)
            scores.append(score)

        return masks, scores

    def segment_grid(
        self, image: np.ndarray, rows: int = 8, cols: int = 12, padding: float = 0.05
    ) -> Tuple[List[np.ndarray], List[str]]:
        """使用网格策略分割规则布局"""
        image = self._preprocess_image(image)
        height, width = image.shape[:2]

        cell_height = height / rows
        cell_width = width / cols

        masks = []
        labels = []

        # 生成行标签 A-H
        row_labels = [chr(65 + i) for i in range(rows)]

        # 遍历每个网格单元
        for r, c in tqdm(
            [(r, c) for r in range(rows) for c in range(cols)],
            desc="网格分割",
            ncols=80,
        ):
            # 计算单元格边界
            pad_y = int(cell_height * padding)
            pad_x = int(cell_width * padding)

            y1 = int(r * cell_height) + pad_y
            y2 = int((r + 1) * cell_height) - pad_y
            x1 = int(c * cell_width) + pad_x
            x2 = int((c + 1) * cell_width) - pad_x

            # 创建边界框
            box = np.array([x1, y1, x2, y2])

            try:
                mask, score = self.segment_with_prompts(image, boxes=box)
                masks.append(mask)
                labels.append(f"{row_labels[r]}{c+1}")
            except Exception as e:
                logging.warning(f"分割网格单元 {row_labels[r]}{c+1} 失败: {e}")
                empty_mask = np.zeros((height, width), dtype=bool)
                masks.append(empty_mask)
                labels.append(f"{row_labels[r]}{c+1}")

        return masks, labels

    def segment_with_prompts(
        self,
        image: np.ndarray,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """使用提示进行分割"""
        image = self._preprocess_image(image)
        self.predictor.set_image(image)

        # 准备输入
        point_coords = np.array(points) if points else None
        point_labels_array = np.array(point_labels) if point_labels else None

        # 执行分割
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels_array,
            box=boxes,
            multimask_output=True,
        )

        # 选择最佳掩码
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]

    def find_diffusion_zone(
        self, image: np.ndarray, colony_mask: np.ndarray, expansion_pixels: int = 15
    ) -> np.ndarray:
        """寻找菌落的扩散区域"""
        kernel = np.ones((3, 3), np.uint8)
        iterations = max(1, expansion_pixels // 3)

        expanded_mask = cv2.dilate(
            colony_mask.astype(np.uint8), kernel, iterations=iterations
        )

        diffusion_mask = expanded_mask - colony_mask.astype(np.uint8)
        return diffusion_mask > 0

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像，确保格式正确"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        return image

    @property
    def is_ready(self) -> bool:
        """检查模型是否准备就绪"""
        return (
            hasattr(self, "sam")
            and hasattr(self, "predictor")
            and hasattr(self, "mask_generator")
        )
