# ============================================================================
# 6. colony_analysis/core/detection.py - 菌落检测器
# ============================================================================

# colony_analysis/core/detection.py - 增量更新版本
# 保留原有的基础函数，只更新和添加需要的部分

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .sam_model import SAMModel
from ..utils.validation import ImageValidator, DataValidator


# ✅ 更新数据类 - 添加新字段到现有的DetectionConfig
@dataclass
class DetectionConfig:
    """检测配置数据类 - 完整版"""
    mode: str = 'auto'
    min_colony_area: int = 500
    max_colony_area: int = 50000
    expand_pixels: int = 2
    merge_overlapping: bool = True
    use_preprocessing: bool = True
    overlap_threshold: float = 0.3
    background_filter: bool = True

    # 混合模式专用参数
    enable_multi_stage: bool = True
    high_quality_threshold: float = 0.8
    supplementary_threshold: float = 0.65
    max_background_ratio: float = 0.2
    edge_contact_limit: float = 0.3
    shape_regularity_min: float = 0.2

    # 去重相关参数
    duplicate_centroid_threshold: float = 50.0  # 中心点距离阈值
    duplicate_overlap_threshold: float = 0.6     # 边界框重叠阈值
    enable_duplicate_merging: bool = False       # 是否启用信息合并
      # 增强功能开关
    enable_adaptive_grid: bool = True      # 启用自适应网格调整
    sort_by_quality: bool = True           # 按质量分数排序结果
    min_quality_score: float = 0.3          # 最低质量分数阈值
  
    # Hybrid模式参数
    min_colonies_expected: int = 30       # 预期最少菌落数
    max_mapping_distance: float = 0.4       # 最大映射距离（相对于孔位大小）
    supplement_score_threshold: float = 0.5 # 补充检测的分数阈值
    edge_margin_ratio: float = 0.08         # 边缘边距比例
  
    # 跨界处理参数
    cross_boundary_overlap_threshold: float = 0.1  # 跨界判定的重叠阈值
    mark_cross_boundary: bool = True              # 是否标记跨界菌落




class ColonyDetector:
    """统一的菌落检测器"""
    # base class for colony detection, integrating SAMModel and configuration management
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
            # 提取所有可用的参数
            for field in DetectionConfig.__dataclass_fields__:
                if hasattr(detection_obj, field):
                    detection_params[field] = getattr(detection_obj, field)

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

    #preprocess_image
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

    #three detection modes
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
    
    def _detect_hybrid_mode(self, img: np.ndarray) -> List[Dict]:
        """改进的混合检测模式 - 集成增强功能"""
        logging.info("使用改进的混合检测模式...")

        # Step 1: 使用auto模式精确检测菌落
        auto_colonies = self._detect_auto_mode_refined(img)
        logging.info(f"Auto检测到 {len(auto_colonies)} 个菌落")

        # Step 2: 创建孔板网格映射
        plate_grid = self._create_plate_grid(img.shape[:2])

        # Step 2.5: 【新增】自适应调整网格（如果启用）
        if hasattr(self.config, 'enable_adaptive_grid') and self.config.enable_adaptive_grid:
            # 先做一次初步映射
            temp_mapped = self._map_colonies_to_wells(
                auto_colonies.copy(), plate_grid)
            # 根据映射结果调整网格
            plate_grid = self._adaptive_grid_adjustment(
                img, plate_grid, temp_mapped)
            logging.info("已根据检测结果调整网格位置")

        # Step 3: 将检测到的菌落映射到最近的孔位
        mapped_colonies = self._map_colonies_to_wells(
            auto_colonies, plate_grid)

        # Step 3.5: 【新增】处理跨界菌落
        mapped_colonies = self._cross_boundary_colony_handling(
            mapped_colonies, plate_grid)

        # Step 4: 补充检测遗漏的孔位
        if len(mapped_colonies) < self.config.min_colonies_expected:
            supplemented = self._supplement_missing_wells(
                img, mapped_colonies, plate_grid)
            mapped_colonies.extend(supplemented)

        # Step 5: 【新增】计算质量分数
        for colony in mapped_colonies:
            self._quality_score_adjustment(colony)

        # Step 6: 【新增】根据质量分数排序（可选）
        if hasattr(self.config, 'sort_by_quality') and self.config.sort_by_quality:
            mapped_colonies.sort(key=lambda x: x.get(
                'quality_score', 0), reverse=True)

        logging.info(f"最终检测到 {len(mapped_colonies)} 个菌落")
        if self.config.enable_multi_stage:
            mapped_colonies = self._remove_duplicates(mapped_colonies)
        # 统计信息
        cross_boundary_count = sum(
            1 for c in mapped_colonies if c.get('cross_boundary', False))
        if cross_boundary_count > 0:
            logging.info(f"其中 {cross_boundary_count} 个菌落跨越孔位边界")

        avg_quality = np.mean([c.get('quality_score', 0.5)
                              for c in mapped_colonies])
        logging.info(f"平均质量分数: {avg_quality:.3f}")

        return mapped_colonies

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

    #Hybird detection methods
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
    
    def _cross_boundary_colony_handling(self, colonies: List[Dict],
                                            grid_info: Dict) -> List[Dict]:
        """
        处理跨越孔位边界的菌落
        
        使用场景：在孔位映射后调用，标记和处理跨界情况
        """
        for colony in colonies:
            bbox = colony['bbox']
            overlapping_wells = []
            overlap_ratios = {}

            for well_id, info in grid_info.items():
                well_bbox = info.get('expected_bbox', info.get('bbox'))
                if not well_bbox:
                    continue

                overlap = self._calculate_bbox_overlap(bbox, well_bbox)
                if overlap > 0.1:  # 10%以上的重叠
                    overlapping_wells.append(well_id)
                    overlap_ratios[well_id] = overlap

            if len(overlapping_wells) > 1:
                # 标记为跨界菌落
                colony['cross_boundary'] = True
                colony['overlapping_wells'] = overlapping_wells
                colony['overlap_ratios'] = overlap_ratios

                # 选择重叠最大的孔位作为主要归属
                if not colony.get('well_position') or colony['well_position'].startswith('unmapped'):
                    primary_well = max(overlap_ratios.items(),
                                       key=lambda x: x[1])[0]
                    colony['well_position'] = primary_well
                    colony['id'] = f"{primary_well}_cross"

                logging.warning(
                    f"检测到跨界菌落: {colony['id']} 跨越 {overlapping_wells}")
            else:
                colony['cross_boundary'] = False

        return colonies

    def _supplement_missing_wells(self, img: np.ndarray, existing: List[Dict],
                                      grid_info: Dict[str, Dict]) -> List[Dict]:
        """补充检测遗漏的孔位"""
        used_wells = {c.get('well_position') for c in existing
                    if c.get('well_position') and not c['well_position'].startswith('unmapped')}
        missing_wells = set(grid_info.keys()) - used_wells

        if not missing_wells or len(missing_wells) > 50:  # 太多空位说明可能有问题
            return []

        logging.info(f"尝试补充检测 {len(missing_wells)} 个空孔位")

        supplemented = []
        for well_id in list(missing_wells)[:20]:  # 最多补充20个
            info = grid_info[well_id]
            bbox = info['expected_bbox']

            try:
                # 在孔位中心使用点提示
                center_y, center_x = info['center']
                points = [[center_x, center_y]]

                mask, score = self.sam_model.segment_with_prompts(
                    img, points=points, point_labels=[1]
                )

                if score > 0.5 and np.sum(mask) > self.config.min_colony_area // 2:
                    colony_data = self._extract_colony_data(
                        img, mask, well_id, 'hybrid_supplement'
                    )

                    if colony_data:
                        colony_data['well_position'] = well_id
                        colony_data['id'] = well_id
                        colony_data['row'] = info['row']
                        colony_data['col'] = info['col']
                        colony_data['sam_score'] = float(score)
                        supplemented.append(colony_data)

            except Exception as e:
                logging.debug(f"补充检测 {well_id} 失败: {e}")
                continue

        logging.info(f"成功补充检测 {len(supplemented)} 个菌落")
        return supplemented

    def _adaptive_grid_adjustment(self, img: np.ndarray, initial_grid: Dict,
                                      detected_colonies: List[Dict]) -> Dict:
        """
        自适应网格调整 - 根据检测结果微调网格位置
        
        使用场景：当检测到的菌落普遍偏离预设网格中心时
        """
        if len(detected_colonies) < 10:  # 样本太少，不调整
            return initial_grid

        # 计算整体偏移
        total_offset_y = 0
        total_offset_x = 0
        valid_mappings = 0

        for colony in detected_colonies:
            if 'well_position' not in colony or colony['well_position'].startswith('unmapped'):
                continue

            well_id = colony['well_position']
            if well_id not in initial_grid:
                continue

            # 计算实际位置与网格中心的偏差
            expected_center = initial_grid[well_id]['center']
            actual_center = colony['centroid']

            offset_y = actual_center[0] - expected_center[0]
            offset_x = actual_center[1] - expected_center[1]

            # 只统计合理范围内的偏移
            if abs(offset_y) < 50 and abs(offset_x) < 50:
                total_offset_y += offset_y
                total_offset_x += offset_x
                valid_mappings += 1

        if valid_mappings < 5:  # 有效映射太少
            return initial_grid

        # 计算平均偏移
        avg_offset_y = total_offset_y / valid_mappings
        avg_offset_x = total_offset_x / valid_mappings

        # 如果偏移显著，调整网格
        if abs(avg_offset_y) > 10 or abs(avg_offset_x) > 10:
            logging.info(
                f"检测到网格偏移: Y={avg_offset_y:.1f}, X={avg_offset_x:.1f}")

            adjusted_grid = {}
            for well_id, info in initial_grid.items():
                adjusted_info = info.copy()
                old_center = info['center']
                adjusted_info['center'] = (
                    old_center[0] + avg_offset_y,
                    old_center[1] + avg_offset_x
                )
                adjusted_grid[well_id] = adjusted_info

            return adjusted_grid

        return initial_grid

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



    #tools and methods
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
    
    def _is_background_region(self, mask: np.ndarray, img: np.ndarray) -> bool:
        """检测是否为背景区域"""
        try:
            # 使用配置中的参数
            h, w = mask.shape
            area = np.sum(mask)
            img_area = h * w

            # 1. 面积检查
            if area > img_area * self.config.max_background_ratio:
                logging.debug(f"背景检测: 面积过大 {area/img_area:.3f}")
                return True

            # 2. 边缘接触检查
            edge_pixels = (np.sum(mask[0, :]) + np.sum(mask[-1, :]) +
                           np.sum(mask[:, 0]) + np.sum(mask[:, -1]))
            edge_ratio = edge_pixels / area if area > 0 else 0

            if edge_ratio > self.config.edge_contact_limit:
                logging.debug(f"背景检测: 边缘接触过多 {edge_ratio:.3f}")
                return True

            # 3. 形状规整度检查（可选）
            if hasattr(self.config, 'shape_regularity_min'):
                regularity = self._calculate_shape_regularity(mask)
                if regularity < self.config.shape_regularity_min:
                    logging.debug(f"背景检测: 形状过于不规则 {regularity:.3f}")
                    return True

            return False

        except Exception as e:
            logging.error(f"背景检测出错: {e}")
            return False

    def _calculate_shape_regularity(self, mask: np.ndarray) -> float:
        """计算形状规整度（圆形度）"""
        try:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return 0

            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)

            if perimeter == 0:
                return 0

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            return min(circularity, 1.0)

        except Exception:
            return 0


    #post_process

    def _post_process_colonies(self, colonies: List[Dict]) -> List[Dict]:
        """后处理菌落列表 - 增强版"""
        if not colonies:
            return colonies

        # 原有的验证逻辑
        valid_colonies = []
        for colony in colonies:
            is_valid, error_msg = DataValidator.validate_colony(colony)
            if is_valid:
                valid_colonies.append(colony)
            else:
                logging.debug(f"移除无效菌落: {error_msg}")

        # 【新增】计算质量分数（如果还没有）
        for colony in valid_colonies:
            if 'quality_score' not in colony:
                self._quality_score_adjustment(colony)

        # 【新增】根据质量分数过滤（可选）
        if hasattr(self.config, 'min_quality_score'):
            min_score = self.config.min_quality_score
            quality_filtered = [c for c in valid_colonies if c.get(
                'quality_score', 0) >= min_score]

            if len(quality_filtered) < len(valid_colonies):
                logging.info(
                    f"质量过滤: {len(valid_colonies)} -> {len(quality_filtered)}")
                valid_colonies = quality_filtered

        # 过滤重叠菌落（使用质量分数改进优先级）
        if self.config.merge_overlapping and len(valid_colonies) > 1:
            valid_colonies = self._filter_overlapping_colonies_by_quality(
                valid_colonies)

        return valid_colonies
    
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
    
    def _filter_overlapping_colonies_by_quality(self, colonies: List[Dict]) -> List[Dict]:
        """根据质量分数过滤重叠菌落"""
        if len(colonies) <= 1:
            return colonies

        # 按质量分数排序，而不是简单按面积
        sorted_colonies = sorted(
            colonies,
            key=lambda x: x.get('quality_score', x.get('sam_score', 0)),
            reverse=True
        )

        filtered_colonies = []
        used_regions = []

        for colony in sorted_colonies:
            bbox = colony['bbox']

            is_overlapping = False
            for used_bbox in used_regions:
                if self._calculate_bbox_overlap(bbox, used_bbox) > self.config.overlap_threshold:
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered_colonies.append(colony)
                used_regions.append(bbox)

        logging.info(f"质量优先重叠过滤：{len(colonies)} -> {len(filtered_colonies)}")
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
        """
        移除重复的菌落 - 用于合并不同检测方法的结果
        
        重复判定标准：
        1. 中心点距离小于阈值
        2. 边界框重叠超过阈值
        3. 优先保留质量分数高的
        """
        if len(colonies) <= 1:
            return colonies

        logging.info(f"去重前: {len(colonies)} 个菌落")

        # 按质量分数排序，优先保留高质量的
        def get_quality_score(colony):
            # SAM分数
            sam_score = colony.get('sam_score', 0.5)

            # 检测方法优先级
            method_priority = {
                'sam_auto_refined': 1.0,
                'sam_auto': 0.9,
                'sam_grid': 0.8,
                'hybrid_supplement': 0.7
            }
            method = colony.get('detection_method', 'unknown')
            method_score = method_priority.get(method, 0.5)

            # 面积合理性（假设理想面积在5000左右）
            area = colony.get('area', 0)
            area_score = 1.0 - abs(area - 5000) / 10000
            area_score = max(0, min(1, area_score))

            # 综合分数
            return sam_score * 0.5 + method_score * 0.3 + area_score * 0.2

        sorted_colonies = sorted(colonies, key=get_quality_score, reverse=True)

        unique_colonies = []

        for i, colony in enumerate(sorted_colonies):
            is_duplicate = False

            # 与已接受的菌落比较
            for accepted in unique_colonies:
                # 检查中心点距离
                if self._check_centroid_distance(colony, accepted):
                    is_duplicate = True
                    logging.debug(
                        f"菌落 {colony.get('id')} 与 {accepted.get('id')} 中心点过近")
                    break

                # 检查边界框重叠
                overlap = self._calculate_bbox_overlap(
                    colony['bbox'], accepted['bbox'])
                if overlap > 0.6:  # 60%重叠认为是重复
                    is_duplicate = True
                    logging.debug(
                        f"菌落 {colony.get('id')} 与 {accepted.get('id')} 重叠 {overlap:.2f}")
                    break

            if not is_duplicate:
                unique_colonies.append(colony)

        logging.info(
            f"去重后: {len(unique_colonies)} 个菌落 (移除 {len(colonies) - len(unique_colonies)} 个)")

        return unique_colonies

    def _check_centroid_distance(self, colony1: Dict, colony2: Dict,
                                 threshold: float = 50.0) -> bool:
        """
        检查两个菌落的中心点距离是否过近
        
        Args:
            colony1, colony2: 菌落数据
            threshold: 距离阈值（像素）
        
        Returns:
            bool: True if too close (likely duplicate)
        """
        if 'centroid' not in colony1 or 'centroid' not in colony2:
            return False

        c1 = colony1['centroid']
        c2 = colony2['centroid']

        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

        return distance < threshold

    def _merge_duplicate_colonies(self, colonies: List[Dict]) -> List[Dict]:
        """
        合并重复菌落的高级版本 - 不仅去重，还可以合并信息
        
        当两个菌落被判定为重复时，可以选择合并它们的信息
        而不是简单地丢弃一个
        """
        if len(colonies) <= 1:
            return colonies

        # 构建重复组
        duplicate_groups = []
        processed = set()

        for i, colony1 in enumerate(colonies):
            if i in processed:
                continue

            group = [colony1]
            processed.add(i)

            for j, colony2 in enumerate(colonies[i+1:], i+1):
                if j in processed:
                    continue

                # 检查是否重复
                centroid_close = self._check_centroid_distance(colony1, colony2)
                bbox_overlap = self._calculate_bbox_overlap(
                    colony1['bbox'], colony2['bbox']
                ) > 0.5

                if centroid_close or bbox_overlap:
                    group.append(colony2)
                    processed.add(j)

            duplicate_groups.append(group)

        # 合并每组重复菌落
        merged_colonies = []

        for group in duplicate_groups:
            if len(group) == 1:
                merged_colonies.append(group[0])
            else:
                # 合并策略：选择最佳的基础菌落，然后补充信息
                best_colony = max(group, key=lambda c: c.get('sam_score', 0))

                # 可以从其他重复菌落中补充信息
                # 例如：如果一个有孔位信息，另一个没有
                for colony in group:
                    if 'well_position' in colony and 'well_position' not in best_colony:
                        best_colony['well_position'] = colony['well_position']
                        best_colony['row'] = colony.get('row')
                        best_colony['column'] = colony.get('column')

                # 记录合并信息
                best_colony['merged_from'] = len(group)
                best_colony['detection_methods'] = list(set(
                    c.get('detection_method', 'unknown') for c in group
                ))

                merged_colonies.append(best_colony)

        logging.info(f"合并重复菌落: {len(colonies)} -> {len(merged_colonies)}")

        return merged_colonies
    
    def _quality_score_adjustment(self, colony: Dict) -> float:
        """
        基于多个因素计算菌落质量分数
        
        使用场景：在最终结果输出前调用，为每个菌落计算综合质量分数
        """
        # 基础SAM分数
        base_score = colony.get('sam_score', 0.5)

        # 位置因素（成功映射到孔位的加分）
        position_bonus = 0
        if 'well_position' in colony and not colony['well_position'].startswith('unmapped'):
            position_bonus = 0.1
            # 如果不是跨界的，再加分
            if not colony.get('cross_boundary', False):
                position_bonus += 0.05

        # 形状因素
        shape_bonus = 0
        if 'features' in colony:
            circularity = colony['features'].get('circularity', 0)
            shape_bonus = circularity * 0.1
        else:
            # 快速计算圆形度
            if 'mask' in colony:
                regularity = self._calculate_shape_regularity(colony['mask'])
                shape_bonus = regularity * 0.1

        # 大小因素
        area = colony.get('area', 0)
        size_bonus = 0
        if 1000 < area < 20000:  # 理想范围
            size_bonus = 0.1
        elif 500 < area < 30000:  # 可接受范围
            size_bonus = 0.05

        # 检测方法因素
        method_bonus = {
            'sam_auto_refined': 0.1,
            'high_quality': 0.1,
            'sam_auto': 0.05,
            'sam_grid': 0.05,
            'hybrid_supplement': 0,
            'supplementary': 0
        }.get(colony.get('detection_method', ''), 0)

        # 计算最终分数
        final_score = base_score + position_bonus + \
            shape_bonus + size_bonus + method_bonus

        # 存储详细评分
        colony['quality_score'] = min(final_score, 1.0)
        colony['quality_details'] = {
            'base_score': base_score,
            'position_bonus': position_bonus,
            'shape_bonus': shape_bonus,
            'size_bonus': size_bonus,
            'method_bonus': method_bonus
        }

        return colony['quality_score']
