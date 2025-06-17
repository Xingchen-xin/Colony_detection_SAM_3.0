# ============================================================================
# 12. colony_analysis/utils/visualization.py - 可视化工具
# ============================================================================

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

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
            # 只取 ROI 对应的掩码，保证尺寸匹配
            roi_mask = mask[minr:maxr, minc:maxc].astype(bool)
            colored_layer[:, :] = (b, g, r)

            # 半透明融合
            alpha = 0.4
            # apply weighted blend to entire ROI, then mask in only the masked pixels
            blended = cv2.addWeighted(roi, 1 - alpha, colored_layer, alpha, 0)
            roi[roi_mask] = blended[roi_mask]

            # 把修改后的 ROI 写回
            annotated_img[minr:maxr, minc:maxc] = roi

            # 轮廓绘制（在 ROI 上绘制再写回）
            contours, _ = cv2.findContours(
            roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
    
    @staticmethod
    def overlay_masks(img_rgb: np.ndarray, masks: List[np.ndarray], output_path: Path, 
                      colony_data: Optional[List[Dict]] = None):
        """在原图上叠加所有掩码"""
        if not masks:
            logging.warning("没有掩码需要可视化")
            return
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建叠加图像
        overlay = img_rgb.copy()
        
        # 为每个掩码生成随机颜色
        colors = []
        for _ in masks:
            colors.append([
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            ])
        
        # 叠加所有掩码
        for i, (mask, color) in enumerate(zip(masks, colors)):
            if mask is None or mask.size == 0:
                continue

            # 如果掩码尺寸与原图不同，尝试根据 bbox 还原到全尺寸
            if mask.shape[:2] != img_rgb.shape[:2]:
                full_mask = np.zeros(img_rgb.shape[:2], dtype=mask.dtype)
                if colony_data and i < len(colony_data) and 'bbox' in colony_data[i]:
                    minr, minc, maxr, maxc = colony_data[i]['bbox']
                    h, w = mask.shape[:2]
                    full_mask[minr:minr+h, minc:minc+w] = mask
                else:
                    full_mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                full_mask = mask

            # 创建彩色掩码
            colored_mask = np.zeros_like(img_rgb)
            for c in range(3):
                colored_mask[:, :, c] = full_mask * color[c]

            # 半透明叠加
            alpha = 0.4
            mask_indices = full_mask > 0
            overlay[mask_indices] = (
                overlay[mask_indices] * (1 - alpha) +
                colored_mask[mask_indices] * alpha
            ).astype(np.uint8)

            # 绘制轮廓
            contours, _ = cv2.findContours(
                full_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)
            
            # 添加标签（如果有菌落数据）
            if colony_data and i < len(colony_data):
                colony = colony_data[i]
                if 'centroid' in colony:
                    cy, cx = colony['centroid']
                    label = colony.get('well_position', colony.get('id', f'C{i}'))
                    cv2.putText(overlay, str(label), 
                              (int(cx-10), int(cy-10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (255, 255, 255), 2)
                    cv2.putText(overlay, str(label), 
                              (int(cx-10), int(cy-10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              color, 1)
        
        # 保存图像
        output_file = output_path / "detection_overlay.jpg"
        success = cv2.imwrite(str(output_file), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        if success:
            logging.info(f"检测结果叠加图已保存: {output_file}")
        else:
            logging.error(f"保存叠加图失败: {output_file}")
        
        # 也保存原图作为参考
        original_file = output_path / "original_image.jpg"
        cv2.imwrite(str(original_file), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        return overlay
    
    def create_annotated_image(self, img_rgb: np.ndarray, colonies: List[Dict], 
                              sample_name: str, orientation: str):
        """创建带注释的图像"""
        annotated = img_rgb.copy()
        height, width = annotated.shape[:2]
        
        # 添加标题
        title = f"{sample_name} - {orientation}"
        cv2.putText(annotated, title, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(annotated, title, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # 添加统计信息
        info_text = f"Total colonies: {len(colonies)}"
        cv2.putText(annotated, info_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, info_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # 标注每个菌落
        for i, colony in enumerate(colonies):
            if 'mask' not in colony:
                continue
                
            mask = colony['mask']
            bbox = colony.get('bbox')
            
            if bbox:
                # 绘制边界框
                minr, minc, maxr, maxc = bbox
                cv2.rectangle(annotated, 
                            (int(minc), int(minr)), 
                            (int(maxc), int(maxr)), 
                            (0, 255, 0), 2)
                
                # 添加标签
                label = colony.get('well_position', colony.get('id', f'C{i}'))
                cv2.putText(annotated, str(label), 
                          (int(minc), int(minr) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (0, 255, 0), 2)
        
        # 保存注释图像
        filename = f"annotated_{orientation}_{sample_name}.png"
        output_path = self.viz_dir / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        logging.info(f"注释图像已保存: {output_path}")
        
        return annotated
    
    def create_summary_plot(self, colonies: List[Dict]):
        """创建统计摘要图"""
        if not colonies:
            logging.warning("没有菌落数据用于创建摘要图")
            return
        
        # 提取数据
        areas = [c.get('area', 0) for c in colonies]
        scores = [c.get('sam_score', 0) for c in colonies if 'sam_score' in c]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 面积分布直方图
        axes[0, 0].hist(areas, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Colony Area Distribution')
        axes[0, 0].set_xlabel('Area (pixels)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. SAM分数分布
        if scores:
            axes[0, 1].hist(scores, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('SAM Score Distribution')
            axes[0, 1].set_xlabel('SAM Score')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No SAM scores available', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. 面积箱线图
        axes[1, 0].boxplot(areas, vert=True)
        axes[1, 0].set_title('Colony Area Box Plot')
        axes[1, 0].set_ylabel('Area (pixels)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 检测方法分布
        methods = {}
        for c in colonies:
            method = c.get('detection_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        if methods:
            axes[1, 1].bar(methods.keys(), methods.values(), alpha=0.7)
            axes[1, 1].set_title('Detection Method Distribution')
            axes[1, 1].set_xlabel('Method')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.viz_dir / 'colony_statistics.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"统计图表已保存: {output_path}")



import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Optional
import random

class ImprovedVisualizer:
    """改进的可视化工具，确保图像正确保存"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"可视化输出目录: {self.viz_dir}")
    
    @staticmethod
    def overlay_masks(img_rgb: np.ndarray, masks: List[np.ndarray], output_path: Path, 
                      colony_data: Optional[List[Dict]] = None):
        """在原图上叠加所有掩码"""
        if not masks:
            logging.warning("没有掩码需要可视化")
            return
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建叠加图像
        overlay = img_rgb.copy()
        
        # 为每个掩码生成随机颜色
        colors = []
        for _ in masks:
            colors.append([
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            ])
        
        # 叠加所有掩码
        for i, (mask, color) in enumerate(zip(masks, colors)):
            if mask is None or mask.size == 0:
                continue

            if mask.shape[:2] != img_rgb.shape[:2]:
                full_mask = np.zeros(img_rgb.shape[:2], dtype=mask.dtype)
                if colony_data and i < len(colony_data) and 'bbox' in colony_data[i]:
                    minr, minc, maxr, maxc = colony_data[i]['bbox']
                    h, w = mask.shape[:2]
                    full_mask[minr:minr+h, minc:minc+w] = mask
                else:
                    full_mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                full_mask = mask

            # 创建彩色掩码
            colored_mask = np.zeros_like(img_rgb)
            for c in range(3):
                colored_mask[:, :, c] = full_mask * color[c]

            # 半透明叠加
            alpha = 0.4
            mask_indices = full_mask > 0
            overlay[mask_indices] = (
                overlay[mask_indices] * (1 - alpha) +
                colored_mask[mask_indices] * alpha
            ).astype(np.uint8)

            # 绘制轮廓
            contours, _ = cv2.findContours(
                full_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)
            
            # 添加标签（如果有菌落数据）
            if colony_data and i < len(colony_data):
                colony = colony_data[i]
                if 'centroid' in colony:
                    cy, cx = colony['centroid']
                    label = colony.get('well_position', colony.get('id', f'C{i}'))
                    cv2.putText(overlay, str(label), 
                              (int(cx-10), int(cy-10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (255, 255, 255), 2)
                    cv2.putText(overlay, str(label), 
                              (int(cx-10), int(cy-10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              color, 1)
        
        # 保存图像
        output_file = output_path / "detection_overlay.jpg"
        success = cv2.imwrite(str(output_file), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        if success:
            logging.info(f"检测结果叠加图已保存: {output_file}")
        else:
            logging.error(f"保存叠加图失败: {output_file}")
        
        # 也保存原图作为参考
        original_file = output_path / "original_image.jpg"
        cv2.imwrite(str(original_file), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        return overlay
    
    def create_annotated_image(self, img_rgb: np.ndarray, colonies: List[Dict], 
                              sample_name: str, orientation: str):
        """创建带注释的图像"""
        annotated = img_rgb.copy()
        height, width = annotated.shape[:2]
        
        # 添加标题
        title = f"{sample_name} - {orientation}"
        cv2.putText(annotated, title, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(annotated, title, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # 添加统计信息
        info_text = f"Total colonies: {len(colonies)}"
        cv2.putText(annotated, info_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, info_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # 标注每个菌落
        for i, colony in enumerate(colonies):
            if 'mask' not in colony:
                continue
                
            mask = colony['mask']
            bbox = colony.get('bbox')
            
            if bbox:
                # 绘制边界框
                minr, minc, maxr, maxc = bbox
                cv2.rectangle(annotated, 
                            (int(minc), int(minr)), 
                            (int(maxc), int(maxr)), 
                            (0, 255, 0), 2)
                
                # 添加标签
                label = colony.get('well_position', colony.get('id', f'C{i}'))
                cv2.putText(annotated, str(label), 
                          (int(minc), int(minr) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (0, 255, 0), 2)
        
        # 保存注释图像
        filename = f"annotated_{orientation}_{sample_name}.png"
        output_path = self.viz_dir / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        logging.info(f"注释图像已保存: {output_path}")
        
        return annotated
    
    def create_summary_plot(self, colonies: List[Dict]):
        """创建统计摘要图"""
        if not colonies:
            logging.warning("没有菌落数据用于创建摘要图")
            return
        
        # 提取数据
        areas = [c.get('area', 0) for c in colonies]
        scores = [c.get('sam_score', 0) for c in colonies if 'sam_score' in c]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 面积分布直方图
        axes[0, 0].hist(areas, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Colony Area Distribution')
        axes[0, 0].set_xlabel('Area (pixels)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. SAM分数分布
        if scores:
            axes[0, 1].hist(scores, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('SAM Score Distribution')
            axes[0, 1].set_xlabel('SAM Score')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No SAM scores available', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. 面积箱线图
        axes[1, 0].boxplot(areas, vert=True)
        axes[1, 0].set_title('Colony Area Box Plot')
        axes[1, 0].set_ylabel('Area (pixels)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 检测方法分布
        methods = {}
        for c in colonies:
            method = c.get('detection_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        if methods:
            axes[1, 1].bar(methods.keys(), methods.values(), alpha=0.7)
            axes[1, 1].set_title('Detection Method Distribution')
            axes[1, 1].set_xlabel('Method')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.viz_dir / 'colony_statistics.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"统计图表已保存: {output_path}")


def save_detection_visualization(img_rgb: np.ndarray, colonies: List[Dict], 
                               output_dir: str, sample_info: Dict):
    """保存检测可视化 - 主函数"""
    viz = ImprovedVisualizer(output_dir)
    
    # 1. 创建掩码叠加图
    masks = [c.get('mask') for c in colonies if 'mask' in c]
    if masks:
        viz.overlay_masks(img_rgb, masks, viz.viz_dir, colonies)
    
    # 2. 创建带注释的图像
    sample_name = sample_info.get('sample_name', 'unknown')
    orientation = sample_info.get('orientation', 'unknown')
    viz.create_annotated_image(img_rgb, colonies, sample_name, orientation)
    
    # 3. 创建统计图表
    viz.create_summary_plot(colonies)
    
    # 4. 保存个体菌落图像（前10个）
    colonies_dir = Path(output_dir) / "colonies"
    colonies_dir.mkdir(exist_ok=True)
    
    for i, colony in enumerate(colonies[:10]):
        if 'img' not in colony:
            continue
            
        colony_id = colony.get('well_position', colony.get('id', f'colony_{i}'))
        filename = f"{colony_id}.jpg"
        output_path = colonies_dir / filename
        
        cv2.imwrite(str(output_path), 
                   cv2.cvtColor(colony['img'], cv2.COLOR_RGB2BGR))
        
    logging.info(f"可视化完成，输出目录: {output_dir}")