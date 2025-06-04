# ============================================================================
# 11. colony_analysis/utils/results.py - 结果管理
# ============================================================================


import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import os
import shutil
from tqdm import tqdm


class ResultManager:
    """统一的结果管理器"""

    def __init__(self, output_dir: str):
        """初始化结果管理器"""
        self.output_dir = Path(output_dir)
        self.directories = self._create_directory_structure()

        logging.info(f"结果管理器初始化完成，输出目录: {self.output_dir}")

    def _create_directory_structure(self) -> Dict[str, Path]:
        """创建输出目录结构"""
        directories = {
            'root': self.output_dir,
            'results': self.output_dir / 'results',
            'colonies': self.output_dir / 'colonies',
            'masks': self.output_dir / 'masks',
            'visualizations': self.output_dir / 'visualizations',
            'debug': self.output_dir / 'debug',
            'reports': self.output_dir / 'reports'
        }

        # 创建所有目录
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories

    def save_all_results(self, colonies: List[Dict], args) -> Dict[str, str]:
        """保存所有结果"""
        saved_files = {}

        try:
            # 1. 保存CSV结果
            csv_path = self.save_csv_results(colonies)
            saved_files['csv'] = str(csv_path)

            # 2. 保存JSON结果（如果是高级分析）
            if getattr(args, 'advanced', False):
                json_path = self.save_json_results(colonies)
                saved_files['json'] = str(json_path)

            # 3. 保存菌落图像
            images_dir = self.save_colony_images(colonies)
            saved_files['images'] = str(images_dir)

            # 4. 保存掩码（如果启用调试）
            if getattr(args, 'debug', False):
                masks_dir = self.save_colony_masks(colonies)
                saved_files['masks'] = str(masks_dir)

                # 将 metabolite 调试图像复制到输出 debug 目录
                metabolite_debug_src = Path("debug_metabolite")
                if metabolite_debug_src.exists() and metabolite_debug_src.is_dir():
                    for f in metabolite_debug_src.iterdir():
                        if f.is_file():
                            shutil.move(str(f), str(self.directories['debug'] / f.name))
                    # 可选：删除源目录
                    try:
                        metabolite_debug_src.rmdir()
                    except OSError:
                        pass

            # 5. 生成分析报告
            report_path = self.generate_analysis_report(colonies, args)
            saved_files['report'] = str(report_path)

            logging.info(f"结果保存完成: {len(saved_files)} 个文件/目录")
            return saved_files

        except Exception as e:
            logging.error(f"保存结果时发生错误: {e}")
            raise

    def save_csv_results(self, colonies: List[Dict]) -> Path:
        """保存CSV格式的结果"""
        rows = []

        for colony in tqdm(colonies, desc="保存CSV结果", ncols=80):
            row = {
                'id': colony.get('id', 'unknown'),
                'well_position': colony.get('well_position', ''),
                'area': float(colony.get('area', 0)),
                'detection_method': colony.get('detection_method', 'unknown'),
                'sam_score': colony.get('sam_score', 0.0),
                'quality_score': colony.get('quality_score', 0),
                'cross_boundary': colony.get('cross_boundary', False),
                'overlapping_wells': ','.join(colony.get('overlapping_wells', []))
            }

            # 添加特征
            features = colony.get('features', {})
            for name, value in features.items():
                row[f'feature_{name}'] = self._safe_convert_value(value)

            # 添加评分
            scores = colony.get('scores', {})
            for name, value in scores.items():
                row[f'score_{name}'] = self._safe_convert_value(value)

            # 添加表型
            phenotype = colony.get('phenotype', {})
            for name, value in phenotype.items():
            # 处理可能的列表类型
                if isinstance(value, list):
                    row[f'phenotype_{name}'] = ', '.join(map(str, value))
                else:
                    row[f'phenotype_{name}'] = str(value)
            rows.append(row)

        # 保存CSV
        df = pd.DataFrame(rows)
        csv_path = self.directories['results'] / 'analysis_results.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')

        logging.info(f"CSV结果已保存: {csv_path}")
        return csv_path

    def convert_to_serializable(self, obj):
        """递归将数据转换为可序列化的格式"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            # 遍历字典并递归处理每个值
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # 遍历列表并递归处理每个元素
            return [self.convert_to_serializable(i) for i in obj]
        else:
            # 对于其他类型，直接返回
            return obj

    def save_json_results(self, colonies):
        """保存JSON格式的详细结果"""
        serializable_data = []

        for colony in tqdm(colonies, desc="保存JSON结果", ncols=80):
            colony_data = {
                'id': colony.get('id', 'unknown'),
                'basic_info': {
                    'area': float(colony.get('area', 0)),
                    'centroid': colony.get('centroid', (0, 0)),
                    'bbox': colony.get('bbox', (0, 0, 0, 0)),
                    'detection_method': colony.get('detection_method', 'unknown')
                },
                'analysis_results': {
                    'features': self._serialize_dict(colony.get('features', {})),
                    'scores': self._serialize_dict(colony.get('scores', {})),
                    'phenotype': colony.get('phenotype', {}),
                    'advanced_features': self._serialize_dict(colony.get('advanced_features', {}))
                },
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'sam_score': colony.get('sam_score', 0.0)
                }
            }
            serializable_data.append(colony_data)

        json_path = self.directories['results'] / 'detailed_results.json'
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"保存 JSON 文件失败: {e}")

        logging.info(f"JSON结果已保存: {json_path}")
        return json_path

    def save_colony_images(self, colonies: List[Dict]) -> Path:
        """保存菌落图像"""
        import cv2

        images_saved = 0

        for colony in tqdm(colonies, desc="保存菌落图像", ncols=80):
            if 'img' not in colony:
                continue

            colony_id = colony.get(
                'well_position') or colony.get('id', 'unknown')

            # 保存原始菌落图像
            if 'img' in colony:
                img_path = self.directories['colonies'] / \
                    f"{colony_id}_original.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(
                    colony['img'], cv2.COLOR_RGB2BGR))
                images_saved += 1

            # 保存掩码应用的图像
            if 'masked_img' in colony:
                masked_path = self.directories['colonies'] / \
                    f"{colony_id}_masked.jpg"
                cv2.imwrite(str(masked_path), cv2.cvtColor(
                    colony['masked_img'], cv2.COLOR_RGB2BGR))
                images_saved += 1

        logging.info(f"已保存 {images_saved} 个菌落图像")
        return self.directories['colonies']

    def save_colony_masks(self, colonies: List[Dict]) -> Path:
        """保存菌落掩码"""
        import cv2

        masks_saved = 0

        for colony in tqdm(colonies, desc="保存菌落掩码", ncols=80):
            if 'mask' not in colony:
                continue

            colony_id = colony.get(
                'well_position') or colony.get('id', 'unknown')

            # 保存二值掩码
            mask_path = self.directories['masks'] / f"{colony_id}_mask.png"
            mask_img = (colony['mask'] * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_img)
            masks_saved += 1

        logging.info(f"已保存 {masks_saved} 个菌落掩码")
        return self.directories['masks']

    def generate_analysis_report(self, colonies: List[Dict], args) -> Path:
        """生成分析报告"""
        report_data = {
            'analysis_summary': self._generate_summary_stats(colonies),
            'detection_info': self._generate_detection_info(colonies, args),
            'phenotype_distribution': self._generate_phenotype_stats(colonies),
            'quality_metrics': self._generate_quality_metrics(colonies)
        }

        # 保存报告
        report_path = self.directories['reports'] / 'analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 也生成文本格式的简要报告
        self._generate_text_report(report_data)

        logging.info(f"分析报告已生成: {report_path}")
        return report_path

    def _generate_summary_stats(self, colonies: List[Dict]) -> Dict:
        """生成统计摘要"""
        if not colonies:
            return {'total_colonies': 0}

        areas = [colony.get('area', 0) for colony in colonies]
        scores = [colony.get('sam_score', 0)
                  for colony in colonies if 'sam_score' in colony]

        return {
            'total_colonies': len(colonies),
            'area_stats': {
                'mean': float(np.mean(areas)),
                'median': float(np.median(areas)),
                'std': float(np.std(areas)),
                'min': float(np.min(areas)),
                'max': float(np.max(areas))
            },
            'detection_quality': {
                'mean_sam_score': float(np.mean(scores)) if scores else 0.0,
                'high_quality_colonies': len([s for s in scores if s > 0.9])
            }
        }

    def _generate_detection_info(self, colonies: List[Dict], args) -> Dict:
        """生成检测信息"""
        detection_methods = {}
        for colony in tqdm(colonies, desc="生成定量报告数据", ncols=80):
            method = colony.get('detection_method', 'unknown')
            detection_methods[method] = detection_methods.get(method, 0) + 1

        return {
            'detection_mode': getattr(args, 'mode', 'unknown'),
            'model_type': getattr(args, 'model', 'unknown'),
            'detection_methods': detection_methods,
            'advanced_analysis': getattr(args, 'advanced', False)
        }

    def _generate_phenotype_stats(self, colonies: List[Dict]) -> Dict:
        """生成表型统计"""
        phenotype_stats = {}

        for colony in colonies:
            phenotype = colony.get('phenotype', {})
            for category, value in phenotype.items():
                if category not in phenotype_stats:
                    phenotype_stats[category] = {}
            # 处理特殊情况：如果值是列表，转换为字符串
            if isinstance(value, list):
                value_key = ', '.join(map(str, value)) if value else 'none'
            else:
                value_key = value

            phenotype_stats[category][value_key] = phenotype_stats[category].get(
                value_key, 0) + 1
        return phenotype_stats

    def _generate_quality_metrics(self, colonies: List[Dict]) -> Dict:
        """生成质量指标"""
        total = len(colonies)
        if total == 0:
            return {}

        with_features = len([c for c in colonies if c.get('features')])
        with_scores = len([c for c in colonies if c.get('scores')])
        with_phenotype = len([c for c in colonies if c.get('phenotype')])

        return {
            'completeness': {
                'features_extracted': with_features / total,
                'scores_calculated': with_scores / total,
                'phenotype_classified': with_phenotype / total
            },
            'data_quality': {
                'total_colonies': total,
                'successful_analysis': with_features
            }
        }

    def _generate_text_report(self, report_data: Dict) -> Path:
        """生成文本格式的报告"""
        lines = []
        lines.append("=== 菌落分析报告 ===")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 基本统计
        summary = report_data.get('analysis_summary', {})
        lines.append(f"检测到的菌落总数: {summary.get('total_colonies', 0)}")

        if 'area_stats' in summary:
            area_stats = summary['area_stats']
            lines.append(f"平均面积: {area_stats['mean']:.2f}")
            lines.append(
                f"面积范围: {area_stats['min']:.2f} - {area_stats['max']:.2f}")

        # 检测信息
        detection_info = report_data.get('detection_info', {})
        lines.append(
            f"检测模式: {detection_info.get('detection_mode', 'unknown')}")
        lines.append(f"SAM模型: {detection_info.get('model_type', 'unknown')}")

        # 表型分布
        phenotype_dist = report_data.get('phenotype_distribution', {})
        if phenotype_dist:
            lines.append("\n=== 表型分布 ===")
            for category, distribution in phenotype_dist.items():
                lines.append(f"{category}:")
                for phenotype, count in distribution.items():
                    lines.append(f"  {phenotype}: {count}")

        # 保存文本报告
        text_path = self.directories['reports'] / 'analysis_report.txt'
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return text_path

    def _serialize_dict(self, data: Dict) -> Dict:
        """序列化字典中的numpy类型"""
        if not isinstance(data, dict):
            return data

        serialized = {}
        for key, value in data.items():
            serialized[key] = self._safe_convert_value(value)

        return serialized

    def _safe_convert_value(self, value: Any) -> Any:
        """安全转换值为可序列化类型"""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.generic):
            return value.item()
        elif isinstance(value, (list, tuple)):
            return [self._safe_convert_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._safe_convert_value(v) for k, v in value.items()}
        else:
            return value
