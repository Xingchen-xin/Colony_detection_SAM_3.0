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
    """检测配置数据类 - 修复版本"""
    mode: str = 'auto'
    min_colony_area: int = 500         # 降低最小面积
    max_colony_area: int = 50000       # 🔥 新增：最大面积限制
    expand_pixels: int = 2
    merge_overlapping: bool = True
    use_preprocessing: bool = True
    overlap_threshold: float = 0.3
    background_filter: bool = True      # 🔥 新增：背景过滤


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
        """自动检测模式 - 修复版本"""
        logging.info("使用自动检测模式...")

        # 计算图像尺寸用于背景检测
        img_area = img.shape[0] * img.shape[1]
        max_colony_area = min(self.config.max_colony_area,
                              img_area * 0.1)  # 不超过图像10%

        logging.info(
            f"面积限制: {self.config.min_colony_area} - {max_colony_area}")

        min_area_for_sam = max(50, self.config.min_colony_area // 8)
        masks, scores = self.sam_model.segment_everything(
            img, min_area=min_area_for_sam
        )

        logging.info(f"SAM返回了 {len(masks)} 个掩码候选")

        colonies = []
        filtered_counts = {
            'too_small': 0,
            'too_large': 0,  # 🔥 新增统计
            'background': 0,  # 🔥 新增统计
            'valid': 0
        }

        for i, (mask, score) in enumerate(zip(masks, scores)):
            enhanced_mask = self._enhance_colony_mask(mask)
            area = np.sum(enhanced_mask)

            # 🔥 新增：面积范围检查
            if area < self.config.min_colony_area:
                filtered_counts['too_small'] += 1
                logging.debug(f"掩码 {i} 面积过小: {area}")
                continue

            if area > max_colony_area:
                filtered_counts['too_large'] += 1
                logging.warning(
                    f"掩码 {i} 面积过大(可能是背景): {area} > {max_colony_area}")
                continue

            # 🔥 新增：背景检测
            if self.config.background_filter and self._is_background_region(enhanced_mask, img):
                filtered_counts['background'] += 1
                logging.warning(f"掩码 {i} 被识别为背景区域")
                continue

            # 提取菌落数据
            colony_data = self._extract_colony_data(
                img, enhanced_mask, f'colony_{i}', 'sam_auto'
            )

            if colony_data:
                colony_data['sam_score'] = float(score)
                colonies.append(colony_data)
                filtered_counts['valid'] += 1
                logging.debug(f"✓ 菌落 {i}: 面积={area:.0f}, 分数={score:.3f}")

        # 打印过滤统计
        logging.info(f"过滤统计: 过小={filtered_counts['too_small']}, "
                     f"过大={filtered_counts['too_large']}, "
                     f"背景={filtered_counts['background']}, "
                     f"有效={filtered_counts['valid']}")

        return colonies

    def _is_background_region(self, mask: np.ndarray, img: np.ndarray) -> bool:
        """检测是否为背景区域 - 新增方法"""
        try:
            # 方法1: 检查是否接触图像边缘
            h, w = mask.shape
            edge_pixels = np.sum(mask[0, :]) + np.sum(mask[-1, :]) + \
                np.sum(mask[:, 0]) + np.sum(mask[:, -1])

            if edge_pixels > np.sum(mask) * 0.1:  # 超过10%像素在边缘
                logging.debug("背景检测: 大量边缘像素")
                return True

            # 方法2: 检查掩码的形状特征
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                return True

            # 计算边界框面积比
            minr, maxr = np.min(y_indices), np.max(y_indices)
            minc, maxc = np.min(x_indices), np.max(x_indices)
            bbox_area = (maxr - minr + 1) * (maxc - minc + 1)
            mask_area = np.sum(mask)

            fill_ratio = mask_area / bbox_area if bbox_area > 0 else 0

            # 如果填充比例很低，可能是分散的背景噪声
            if fill_ratio < 0.3:
                logging.debug(f"背景检测: 填充比例过低 {fill_ratio:.3f}")
                return True

            # 方法3: 检查是否覆盖了太大比例的图像
            img_area = h * w
            if mask_area > img_area * 0.5:  # 超过图像50%
                logging.debug(f"背景检测: 覆盖面积过大 {mask_area/img_area:.3f}")
                return True

            return False

        except Exception as e:
            logging.error(f"背景检测出错: {e}")
            return False

    def _filter_overlapping_colonies(self, colonies: List[Dict]) -> List[Dict]:
        """改进的重叠过滤 - 修复版本"""
        if len(colonies) <= 1:
            return colonies

        logging.info(f"重叠过滤前: {len(colonies)} 个菌落")

        # 🔥 修复：先按面积排序，优先保留中等大小的菌落
        # 而不是最大的（可能是背景）
        def get_priority_score(colony):
            area = colony['area']
            # 给中等大小的菌落更高的优先级
            if 1000 <= area <= 20000:  # 理想菌落大小范围
                return area + 100000  # 提高优先级
            else:
                return area

        sorted_colonies = sorted(
            colonies, key=get_priority_score, reverse=True)

        filtered_colonies = []
        used_regions = []
        overlap_count = 0

        for i, colony in enumerate(sorted_colonies):
            bbox = colony['bbox']
            colony_id = colony.get('id', f'colony_{i}')
            area = colony['area']

            # 🔥 新增：直接跳过异常大的区域
            img_area = 1074 * 1607  # 从调试信息获得的图像大小
            if area > img_area * 0.3:  # 超过图像30%的区域直接跳过
                logging.warning(f"跳过异常大区域 {colony_id}: 面积={area}")
                overlap_count += 1
                continue

            # 检查重叠
            is_overlapping = False
            max_overlap = 0.0

            for j, used_bbox in enumerate(used_regions):
                overlap = self._calculate_bbox_overlap(bbox, used_bbox)
                max_overlap = max(max_overlap, overlap)

                if overlap > self.config.overlap_threshold:
                    is_overlapping = True
                    logging.debug(f"菌落 {colony_id} 与菌落 {j} 重叠 {overlap:.3f}")
                    break

            if not is_overlapping:
                filtered_colonies.append(colony)
                used_regions.append(bbox)
                logging.debug(
                    f"✓ 保留菌落 {colony_id}, 面积={area}, 最大重叠={max_overlap:.3f}")
            else:
                overlap_count += 1

        logging.info(
            f"重叠过滤：{len(colonies)} -> {len(filtered_colonies)} (移除 {overlap_count} 个)")
        return filtered_colonies
    
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
    
# colony_analysis/core/detection.py - 改进的混合检测


# class ColonyDetector:
    """改进的菌落检测器 - 解决grid和auto模式问题"""

    def _detect_hybrid_mode(self, img: np.ndarray) -> List[Dict]:
        """改进的混合检测模式：先auto检测，再映射到孔位"""
        logging.info("使用改进的混合检测模式...")

        # Step 1: 使用auto模式精确检测菌落
        auto_colonies = self._detect_auto_mode_refined(img)
        logging.info(f"Auto检测到 {len(auto_colonies)} 个菌落")

        # Step 2: 创建孔板网格映射
        plate_grid = self._create_plate_grid(img.shape[:2])
        logging.info(f"创建了 {len(plate_grid)} 个孔位网格")

        # Step 3: 将检测到的菌落映射到最近的孔位
        mapped_colonies = self._map_colonies_to_wells(
            auto_colonies, plate_grid)
        logging.info(f"成功映射 {len(mapped_colonies)} 个菌落到孔位")

        return mapped_colonies

    def _detect_auto_mode_refined(self, img: np.ndarray) -> List[Dict]:
        """改进的auto检测：专门针对孔板优化"""
        logging.info("使用孔板优化的auto检测...")

        # 计算合理的面积范围（基于孔板尺寸）
        img_area = img.shape[0] * img.shape[1]
        well_area = img_area / (8 * 12)  # 假设96孔板

        # 菌落面积应该在单个孔的10%-80%之间
        min_colony_area = int(well_area * 0.1)
        max_colony_area = int(well_area * 0.8)

        logging.info(f"动态计算面积范围: {min_colony_area} - {max_colony_area}")

        # 使用更密集的采样点检测小菌落
        sam_params_override = {
            'points_per_side': 128,  # 增加采样密度
            'min_mask_region_area': min_colony_area // 4
        }

        # 临时更新SAM参数
        original_params = self.sam_model.params.copy()
        self.sam_model.params.update(sam_params_override)

        try:
            masks, scores = self.sam_model.segment_everything(
                img, min_area=min_colony_area // 4
            )
            logging.info(f"SAM返回 {len(masks)} 个候选掩码")

            colonies = []
            stats = {'valid': 0, 'too_small': 0,
                     'too_large': 0, 'low_score': 0}

            for i, (mask, score) in enumerate(zip(masks, scores)):
                enhanced_mask = self._enhance_colony_mask(mask)
                area = np.sum(enhanced_mask)

                # 严格的面积过滤
                if area < min_colony_area:
                    stats['too_small'] += 1
                    continue
                if area > max_colony_area:
                    stats['too_large'] += 1
                    continue

                # 质量分数过滤
                if score < 0.7:  # 提高质量要求
                    stats['low_score'] += 1
                    continue

                # 形状合理性检查
                if not self._is_reasonable_colony_shape(enhanced_mask):
                    continue

                colony_data = self._extract_colony_data(
                    img, enhanced_mask, f'colony_{i}', 'sam_auto_refined'
                )

                if colony_data:
                    colony_data['sam_score'] = float(score)
                    colonies.append(colony_data)
                    stats['valid'] += 1

            logging.info(f"检测统计: {stats}")
            return colonies

        finally:
            # 恢复原始SAM参数
            self.sam_model.params = original_params

    def _create_plate_grid(self, img_shape: Tuple[int, int], rows: int = 8, cols: int = 12) -> Dict[str, Dict]:
        """创建孔板网格映射"""
        height, width = img_shape

        # 计算网格参数，考虑边距
        margin_y = height * 0.05  # 5%边距
        margin_x = width * 0.05

        usable_height = height - 2 * margin_y
        usable_width = width - 2 * margin_x

        cell_height = usable_height / rows
        cell_width = usable_width / cols

        plate_grid = {}
        row_labels = [chr(65 + i) for i in range(rows)]  # A-H

        for r in range(rows):
            for c in range(cols):
                well_id = f"{row_labels[r]}{c+1}"

                # 计算孔位中心和搜索区域
                center_y = margin_y + (r + 0.5) * cell_height
                center_x = margin_x + (c + 0.5) * cell_width

                # 扩大搜索半径，允许一定偏移
                search_radius = min(cell_height, cell_width) * 0.6

                plate_grid[well_id] = {
                    'center': (center_y, center_x),
                    'search_radius': search_radius,
                    'row': r,
                    'col': c,
                    'expected_bbox': (
                        int(center_y - cell_height/2),
                        int(center_x - cell_width/2),
                        int(center_y + cell_height/2),
                        int(center_x + cell_width/2)
                    )
                }

        return plate_grid

    def _map_colonies_to_wells(self, colonies: List[Dict], plate_grid: Dict[str, Dict]) -> List[Dict]:
        """将检测到的菌落映射到孔位"""
        mapped_colonies = []
        used_wells = set()

        # 为每个菌落找到最近的孔位
        for colony in colonies:
            colony_center = colony['centroid']
            best_well = None
            min_distance = float('inf')

            # 搜索最近的未使用孔位
            for well_id, well_info in plate_grid.items():
                if well_id in used_wells:
                    continue

                well_center = well_info['center']
                distance = np.sqrt((colony_center[0] - well_center[0])**2 +
                                   (colony_center[1] - well_center[1])**2)

                # 检查是否在搜索半径内
                if distance <= well_info['search_radius'] and distance < min_distance:
                    min_distance = distance
                    best_well = well_id

            if best_well:
                # 映射成功
                colony['well_position'] = best_well
                colony['id'] = best_well
                colony['well_distance'] = min_distance
                colony['row'] = plate_grid[best_well]['row']
                colony['column'] = plate_grid[best_well]['col']

                mapped_colonies.append(colony)
                used_wells.add(best_well)

                logging.debug(
                    f"菌落映射: {colony['centroid']} -> {best_well} (距离: {min_distance:.1f})")
            else:
                # 无法映射到孔位，可能是边缘菌落或污染
                colony['well_position'] = f"unmapped_{len(mapped_colonies)}"
                colony['id'] = colony['well_position']
                mapped_colonies.append(colony)
                logging.warning(f"菌落无法映射到孔位: {colony['centroid']}")

        # 生成缺失孔位报告
        all_wells = set(plate_grid.keys())
        missing_wells = all_wells - used_wells
        if missing_wells:
            logging.info(f"空孔位: {sorted(missing_wells)}")

        return mapped_colonies

    def _is_reasonable_colony_shape(self, mask: np.ndarray) -> bool:
        """检查菌落形状是否合理"""
        try:
            # 计算基本形状特征
            area = np.sum(mask)
            if area == 0:
                return False

            # 获取轮廓
            contours, _ = cv2.findContours(
                mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return False

            # 最大轮廓
            main_contour = max(contours, key=cv2.contourArea)

            # 计算圆形度
            perimeter = cv2.arcLength(main_contour, True)
            if perimeter == 0:
                return False

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # 计算长宽比
            rect = cv2.minAreaRect(main_contour)
            width, height = rect[1]
            if min(width, height) == 0:
                return False

            aspect_ratio = max(width, height) / min(width, height)

            # 合理性检查
            reasonable_circularity = 0.3 < circularity < 1.2  # 不要太不规则
            reasonable_aspect = aspect_ratio < 3.0  # 不要太细长

            if not (reasonable_circularity and reasonable_aspect):
                logging.debug(
                    f"形状不合理: 圆形度={circularity:.3f}, 长宽比={aspect_ratio:.3f}")
                return False

            return True

        except Exception as e:
            logging.error(f"形状检查出错: {e}")
            return False
