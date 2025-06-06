# ============================================================================
# 12. colony_analysis/utils/visualization.py - 可视化工具
# ============================================================================

import logging
import random
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """统一的可视化工具"""

    def __init__(self, output_dir: str):
        """初始化可视化工具"""
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def _put_readable_text(
        self, img: np.ndarray, text: str, position: tuple, base_color: tuple
    ):
        """在图像上绘制带黑色边框的文本以增强可读性"""
        # 根据底色亮度选择白色或黑色文字
        brightness = (
            0.299 * base_color[2] + 0.587 * base_color[1] + 0.114 * base_color[0]
        )
        text_color = (0, 0, 0) if brightness > 186 else (255, 255, 255)
        cv2.putText(
            img,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            1,
            cv2.LINE_AA,
        )

    def create_debug_visualizations(
        self, original_img: np.ndarray, colonies: List[Dict]
    ):
        """创建调试可视化"""
        try:
            # 1. 创建检测结果概览
            self.create_detection_overview(original_img, colonies)

            # 2. 创建个体菌落可视化
            self.create_individual_visualizations(colonies)

            # 3. 创建统计图表
            self.create_statistics_plots(colonies)

            logging.info(f"调试可视化已生成: {self.viz_dir}")

        except Exception as e:
            logging.error(f"生成可视化时出错: {e}")

    def create_detection_overview(self, original_img: np.ndarray, colonies: List[Dict]):
        """创建检测结果概览"""
        if not colonies:
            return

        # 复制一份原图用于标注
        annotated_img = original_img.copy()

        # 为每个 colony 随机生成一种颜色（BGR）
        colors = []
        for _ in colonies:
            colors.append(
                (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255),
                )
            )

        for i, colony in enumerate(colonies):
            mask = colony.get("mask", None)
            if mask is None:
                # 回退到绘制边界框
                bbox = colony.get("bbox", None)
                if bbox is None:
                    continue
                minr, minc, maxr, maxc = bbox
                cv2.rectangle(
                    annotated_img,
                    (int(minc), int(minr)),
                    (int(maxc), int(maxr)),
                    colors[i],
                    2,
                )
                label = colony.get("well_position") or colony.get("id", f"C{i}")
                self._put_readable_text(
                    annotated_img, str(label), (int(minc), int(minr) - 10), colors[i]
                )
                continue

            # ------------------- 使用局部 ROI，避免尺寸不一致 -------------------
            # bbox 定位
            bbox = colony.get("bbox", None)
            if bbox is None:
                continue
            minr, minc, maxr, maxc = bbox
            roi = annotated_img[minr:maxr, minc:maxc]

            # 生成与 ROI 同尺寸的彩色层
            b, g, r = colors[i]
            colored_layer = np.zeros_like(roi, dtype=np.uint8)
            colored_layer[:, :] = (b, g, r)

            # 半透明融合
            alpha = 0.4
            mask_bool = mask.astype(bool)
            roi[mask_bool] = cv2.addWeighted(
                roi[mask_bool], 1 - alpha, colored_layer[mask_bool], alpha, 0
            )

            # 把修改后的 ROI 写回
            annotated_img[minr:maxr, minc:maxc] = roi

            # 轮廓绘制（在 ROI 上绘制再写回）
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                annotated_img[minr:maxr, minc:maxc], contours, -1, colors[i], 2
            )

            # 计算质心（局部坐标 -> 全局坐标）
            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cX_local = int(M["m10"] / M["m00"])
                cY_local = int(M["m01"] / M["m00"])
                cX, cY = minc + cX_local, minr + cY_local
            else:
                # fallback to bbox中心
                cY, cX = (minr + maxr) // 2, (minc + maxc) // 2

            label = colony.get("well_position") or colony.get("id", f"C{i}")
            self._put_readable_text(annotated_img, str(label), (cX, cY - 10), colors[i])
            # ------------------- 替换结束 -------------------

        # 保存图像
        overview_path = self.viz_dir / "detection_overview.jpg"
        cv2.imwrite(str(overview_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

    def create_individual_visualizations(self, colonies: List[Dict]):
        """为每个菌落创建个体可视化"""
        individual_dir = self.viz_dir / "individual_colonies"
        individual_dir.mkdir(exist_ok=True)

        for colony in colonies:
            if "img" not in colony or "mask" not in colony:
                continue

            colony_id = colony.get("well_position") or colony.get("id", "unknown")

            # 创建多面板图像
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 原始图像
            axes[0].imshow(colony["img"])
            axes[0].set_title("Original")
            axes[0].axis("off")

            # 掩码
            axes[1].imshow(colony["mask"], cmap="gray")
            axes[1].set_title("Mask")
            axes[1].axis("off")

            # 叠加图像
            overlay = colony["img"].copy()
            overlay[colony["mask"] > 0] = (
                overlay[colony["mask"] > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
            )
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.suptitle(f"Colony {colony_id}")
            plt.tight_layout()

            # 保存
            fig_path = individual_dir / f"{colony_id}_analysis.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()

    def create_statistics_plots(self, colonies: List[Dict]):
        """创建统计图表"""
        if not colonies:
            return

        # 收集统计数据
        areas = [colony.get("area", 0) for colony in colonies]
        sam_scores = [
            colony.get("sam_score", 0) for colony in colonies if "sam_score" in colony
        ]

        # 创建统计图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 面积分布
        axes[0, 0].hist(areas, bins=20, alpha=0.7)
        axes[0, 0].set_title("Area Distribution")
        axes[0, 0].set_xlabel("Area (pixels)")
        axes[0, 0].set_ylabel("Count")

        # SAM分数分布
        if sam_scores:
            axes[0, 1].hist(sam_scores, bins=20, alpha=0.7)
            axes[0, 1].set_title("SAM Score Distribution")
            axes[0, 1].set_xlabel("SAM Score")
            axes[0, 1].set_ylabel("Count")

        # 面积vs分数散点图
        if sam_scores and len(sam_scores) == len(areas):
            axes[1, 0].scatter(areas, sam_scores, alpha=0.6)
            axes[1, 0].set_title("Area vs SAM Score")
            axes[1, 0].set_xlabel("Area (pixels)")
            axes[1, 0].set_ylabel("SAM Score")

        # 检测方法分布
        methods = [colony.get("detection_method", "unknown") for colony in colonies]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1

        if method_counts:
            axes[1, 1].bar(method_counts.keys(), method_counts.values())
            axes[1, 1].set_title("Detection Method Distribution")
            axes[1, 1].set_ylabel("Count")
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # 保存
        stats_path = self.viz_dir / "statistics.png"
        plt.savefig(stats_path, dpi=150, bbox_inches="tight")
        plt.close()
