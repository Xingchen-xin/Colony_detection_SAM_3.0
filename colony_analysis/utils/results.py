# ============================================================================
# 11. colony_analysis/utils/results.py - 结果管理
# ============================================================================


import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
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
            "root": self.output_dir,
            "results": self.output_dir / "results",
            "colonies": self.output_dir / "colonies",
            "masks": self.output_dir / "masks",
            "visualizations": self.output_dir / "visualizations",
            "debug": self.output_dir / "debug",
            "reports": self.output_dir / "reports",
        }

        # 创建所有目录
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories

    def save_all_results(self, colonies: List[Dict], args) -> Dict[str, str]:
        """保存所有结果 - 修复版本"""
        saved_files = {}

        try:
            # 1. 保存CSV结果
            csv_path = self.save_csv_results(colonies)
            saved_files["csv"] = str(csv_path)

            # 2. 保存JSON结果（如果是高级分析）
            if getattr(args, "advanced", False):
                try:
                    json_path = self.save_json_results(colonies)
                    saved_files["json"] = str(json_path)
                except Exception as e:
                    logging.error(f"保存JSON结果失败: {e}")

            # 3. 保存菌落图像
            try:
                images_dir = self.save_colony_images(colonies)
                saved_files["images"] = str(images_dir)
            except Exception as e:
                logging.error(f"保存菌落图像失败: {e}")

            # 4. 保存掩码（如果启用调试）
            if getattr(args, "debug", False):
                try:
                    masks_dir = self.save_colony_masks(colonies)
                    saved_files["masks"] = str(masks_dir)

                    # 将 metabolite 调试图像复制到输出 debug 目录
                    metabolite_debug_src = Path("debug_metabolite")
                    if metabolite_debug_src.exists() and metabolite_debug_src.is_dir():
                        for f in metabolite_debug_src.iterdir():
                            if f.is_file():
                                try:
                                    shutil.move(str(f), str(self.directories["debug"] / f.name))
                                except Exception as e:
                                    logging.warning(f"移动调试文件失败: {f.name}, 错误: {e}")
                        # 可选：删除源目录
                        try:
                            metabolite_debug_src.rmdir()
                        except OSError:
                            pass
                except Exception as e:
                    logging.error(f"保存掩码失败: {e}")

            # 5. 生成分析报告
            try:
                report_path = self.generate_analysis_report(colonies, args)
                saved_files["report"] = str(report_path)
            except Exception as e:
                logging.error(f"生成分析报告失败: {e}")
                # 创建最小报告
                try:
                    minimal_report_path = self.directories["reports"] / "minimal_report.txt"
                    with open(minimal_report_path, "w", encoding="utf-8") as f:
                        f.write(f"分析完成\n")
                        f.write(f"总菌落数: {len(colonies)}\n")
                        f.write(f"报告生成错误: {e}\n")
                    saved_files["report"] = str(minimal_report_path)
                except Exception:
                    pass

            logging.info(f"结果保存完成: {len(saved_files)} 个文件/目录")
            return saved_files

        except Exception as e:
            logging.error(f"保存结果时发生错误: {e}")
            # 确保至少返回部分结果
            return saved_files


    def save_csv_results(self, colonies: List[Dict]) -> Path:
        """保存CSV格式的结果 - 修复版本"""
        rows = []

        for colony in tqdm(colonies, desc="保存CSV结果", ncols=80):
            try:
                row = {
                    "id": colony.get("id", "unknown"),
                    "well_position": colony.get("well_position", ""),
                    "area": float(colony.get("area", 0)),
                    "detection_method": colony.get("detection_method", "unknown"),
                    "sam_score": float(colony.get("sam_score", 0.0)),
                    "quality_score": float(colony.get("quality_score", 0)),
                    "cross_boundary": bool(colony.get("cross_boundary", False)),
                    "overlapping_wells": ",".join(colony.get("overlapping_wells", [])),
                }

                # 安全地添加特征
                features = colony.get("features", {})
                if isinstance(features, dict):
                    for name, value in features.items():
                        try:
                            row[f"feature_{name}"] = self._safe_convert_value(value)
                        except Exception as e:
                            logging.warning(f"转换特征 {name} 失败: {e}")
                            row[f"feature_{name}"] = str(value) if value is not None else ""

                # 安全地添加评分
                scores = colony.get("scores", {})
                if isinstance(scores, dict):
                    for name, value in scores.items():
                        try:
                            row[f"score_{name}"] = self._safe_convert_value(value)
                        except Exception as e:
                            logging.warning(f"转换评分 {name} 失败: {e}")
                            row[f"score_{name}"] = str(value) if value is not None else ""

                # 安全地添加表型
                phenotype = colony.get("phenotype", {})
                if isinstance(phenotype, dict):
                    for name, value in phenotype.items():
                        try:
                            # 处理可能的列表类型
                            if isinstance(value, list):
                                row[f"phenotype_{name}"] = ", ".join(map(str, value)) if value else ""
                            elif value is None:
                                row[f"phenotype_{name}"] = ""
                            else:
                                row[f"phenotype_{name}"] = str(value)
                        except Exception as e:
                            logging.warning(f"转换表型 {name} 失败: {e}")
                            row[f"phenotype_{name}"] = str(value) if value is not None else ""
                
                rows.append(row)
                
            except Exception as e:
                logging.error(f"处理菌落 {colony.get('id', 'unknown')} 时出错: {e}")
                # 添加最小的行数据，确保CSV生成不会完全失败
                rows.append({
                    "id": colony.get("id", "unknown"),
                    "well_position": colony.get("well_position", ""),
                    "area": 0.0,
                    "detection_method": "error",
                    "sam_score": 0.0,
                    "quality_score": 0.0,
                    "cross_boundary": False,
                    "overlapping_wells": "",
                    "error": str(e)
                })

        # 保存CSV
        try:
            if not rows:
                # 如果没有数据，创建空的DataFrame
                rows = [{
                    "id": "no_data",
                    "well_position": "",
                    "area": 0.0,
                    "detection_method": "none",
                    "note": "没有菌落数据"
                }]
            
            df = pd.DataFrame(rows)
            csv_path = self.directories["results"] / "analysis_results.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logging.info(f"CSV结果已保存: {csv_path}")
            return csv_path
            
        except Exception as e:
            logging.error(f"保存CSV失败: {e}")
            # 创建最基本的CSV文件
            csv_path = self.directories["results"] / "analysis_results.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("id,error\n")
                f.write(f"failed,{str(e)}\n")
            return csv_path


    def _safe_convert_value(self, value: Any) -> Any:
        """安全转换值为可序列化类型 - 增强版本"""
        try:
            if value is None:
                return None
            elif isinstance(value, np.ndarray):
                if value.size == 0:
                    return []
                return value.tolist()
            elif isinstance(value, np.generic):
                return value.item()
            elif isinstance(value, (list, tuple)):
                return [self._safe_convert_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): self._safe_convert_value(v) for k, v in value.items()}
            elif isinstance(value, (int, float)):
                # 处理特殊数值
                if np.isnan(value) or np.isinf(value):
                    return 0.0
                return float(value)
            elif isinstance(value, (str, bool)):
                return value
            else:
                # 对于未知类型，尝试转换为字符串
                return str(value)
        except Exception as e:
            logging.warning(f"值转换失败: {value} (类型: {type(value)}), 错误: {e}")
            return str(value) if value is not None else None

    def save_json_results(self, colonies):
        """保存JSON格式的详细结果"""
        serializable_data = []

        for colony in tqdm(colonies, desc="保存JSON结果", ncols=80):
            colony_data = {
                "id": colony.get("id", "unknown"),
                "basic_info": {
                    "area": float(colony.get("area", 0)),
                    "centroid": colony.get("centroid", (0, 0)),
                    "bbox": colony.get("bbox", (0, 0, 0, 0)),
                    "detection_method": colony.get("detection_method", "unknown"),
                },
                "analysis_results": {
                    "features": self._serialize_dict(colony.get("features", {})),
                    "scores": self._serialize_dict(colony.get("scores", {})),
                    "phenotype": colony.get("phenotype", {}),
                    "advanced_features": self._serialize_dict(
                        colony.get("advanced_features", {})
                    ),
                },
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "sam_score": colony.get("sam_score", 0.0),
                },
            }
            serializable_data.append(colony_data)

        json_path = self.directories["results"] / "detailed_results.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
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
            if "img" not in colony:
                continue

            colony_id = colony.get("well_position") or colony.get("id", "unknown")

            # 保存原始菌落图像
            if "img" in colony:
                img_path = self.directories["colonies"] / f"{colony_id}_original.jpg"
                cv2.imwrite(
                    str(img_path), cv2.cvtColor(colony["img"], cv2.COLOR_RGB2BGR)
                )
                images_saved += 1

            # 保存掩码应用的图像
            if "masked_img" in colony:
                masked_path = self.directories["colonies"] / f"{colony_id}_masked.jpg"
                cv2.imwrite(
                    str(masked_path),
                    cv2.cvtColor(colony["masked_img"], cv2.COLOR_RGB2BGR),
                )
                images_saved += 1

        logging.info(f"已保存 {images_saved} 个菌落图像")
        return self.directories["colonies"]

    def save_colony_masks(self, colonies: List[Dict]) -> Path:
        """保存菌落掩码"""
        import cv2

        masks_saved = 0

        for colony in tqdm(colonies, desc="保存菌落掩码", ncols=80):
            if "mask" not in colony:
                continue

            colony_id = colony.get("well_position") or colony.get("id", "unknown")

            # 保存二值掩码
            mask_path = self.directories["masks"] / f"{colony_id}_mask.png"
            mask_img = (colony["mask"] * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_img)
            masks_saved += 1

        logging.info(f"已保存 {masks_saved} 个菌落掩码")
        return self.directories["masks"]

    def generate_analysis_report(self, colonies: List[Dict], args) -> Path:
        """生成分析报告"""
        report_data = {
            "analysis_summary": self._generate_summary_stats(colonies),
            "detection_info": self._generate_detection_info(colonies, args),
            "phenotype_distribution": self._generate_phenotype_stats(colonies),
            "quality_metrics": self._generate_quality_metrics(colonies),
        }

        # 保存报告
        report_path = self.directories["reports"] / "analysis_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 也生成文本格式的简要报告
        self._generate_text_report(report_data)

        logging.info(f"分析报告已生成: {report_path}")
        return report_path

    def _generate_summary_stats(self, colonies: List[Dict]) -> Dict:
        """生成统计摘要"""
        if not colonies:
            return {"total_colonies": 0}

        areas = [colony.get("area", 0) for colony in colonies]
        scores = [
            colony.get("sam_score", 0) for colony in colonies if "sam_score" in colony
        ]

        return {
            "total_colonies": len(colonies),
            "area_stats": {
                "mean": float(np.mean(areas)),
                "median": float(np.median(areas)),
                "std": float(np.std(areas)),
                "min": float(np.min(areas)),
                "max": float(np.max(areas)),
            },
            "detection_quality": {
                "mean_sam_score": float(np.mean(scores)) if scores else 0.0,
                "high_quality_colonies": len([s for s in scores if s > 0.9]),
            },
        }

    def _generate_detection_info(self, colonies: List[Dict], args) -> Dict:
        """生成检测信息"""
        detection_methods = {}
        for colony in tqdm(colonies, desc="生成定量报告数据", ncols=80):
            method = colony.get("detection_method", "unknown")
            detection_methods[method] = detection_methods.get(method, 0) + 1

        return {
            "detection_mode": getattr(args, "mode", "unknown"),
            "model_type": getattr(args, "model", "unknown"),
            "detection_methods": detection_methods,
            "advanced_analysis": getattr(args, "advanced", False),
        }

    def _generate_phenotype_stats(self, colonies: List[Dict]) -> Dict:
        """生成表型统计"""
        phenotype_stats = {}

        for colony in colonies:
            phenotype = colony.get("phenotype", {})
            
            # 确保 phenotype 是字典类型
            if not isinstance(phenotype, dict):
                logging.warning(f"菌落 {colony.get('id', 'unknown')} 的表型数据不是字典格式")
                continue
                
            for category, value in phenotype.items():
                if category not in phenotype_stats:
                    phenotype_stats[category] = {}
                
                # 🔥 修复：将处理 value 的逻辑移到循环内部
                try:
                    # 处理特殊情况：如果值是列表，转换为字符串
                    if isinstance(value, list):
                        value_key = ", ".join(map(str, value)) if value else "none"
                    elif value is None:
                        value_key = "none"
                    else:
                        value_key = str(value)

                    phenotype_stats[category][value_key] = (
                        phenotype_stats[category].get(value_key, 0) + 1
                    )
                except Exception as e:
                    logging.warning(f"处理表型数据时出错: category={category}, value={value}, error={e}")
                    # 使用安全的默认值
                    value_key = "error"
                    phenotype_stats[category][value_key] = (
                        phenotype_stats[category].get(value_key, 0) + 1
                    )
        
        return phenotype_stats

    def _generate_quality_metrics(self, colonies: List[Dict]) -> Dict:
        """生成质量指标"""
        total = len(colonies)
        if total == 0:
            return {}

        with_features = len([c for c in colonies if c.get("features")])
        with_scores = len([c for c in colonies if c.get("scores")])
        with_phenotype = len([c for c in colonies if c.get("phenotype")])

        return {
            "completeness": {
                "features_extracted": with_features / total,
                "scores_calculated": with_scores / total,
                "phenotype_classified": with_phenotype / total,
            },
            "data_quality": {
                "total_colonies": total,
                "successful_analysis": with_features,
            },
        }

    def _generate_text_report(self, report_data: Dict) -> Path:
        """生成文本格式的报告"""
        lines = []
        lines.append("=== 菌落分析报告 ===")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 基本统计
        summary = report_data.get("analysis_summary", {})
        lines.append(f"检测到的菌落总数: {summary.get('total_colonies', 0)}")

        if "area_stats" in summary:
            area_stats = summary["area_stats"]
            lines.append(f"平均面积: {area_stats['mean']:.2f}")
            lines.append(f"面积范围: {area_stats['min']:.2f} - {area_stats['max']:.2f}")

        # 检测信息
        detection_info = report_data.get("detection_info", {})
        lines.append(f"检测模式: {detection_info.get('detection_mode', 'unknown')}")
        lines.append(f"SAM模型: {detection_info.get('model_type', 'unknown')}")

        # 表型分布
        phenotype_dist = report_data.get("phenotype_distribution", {})
        if phenotype_dist:
            lines.append("\n=== 表型分布 ===")
            for category, distribution in phenotype_dist.items():
                lines.append(f"{category}:")
                for phenotype, count in distribution.items():
                    lines.append(f"  {phenotype}: {count}")

        # 保存文本报告
        text_path = self.directories["reports"] / "analysis_report.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

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
# ============================================================================
# 修复 colony_analysis/utils/results.py 中的变量作用域错误
# ============================================================================

# 问题描述：
# 在 _generate_phenotype_stats 方法中，由于缩进错误导致 value 变量
# 在循环外部被访问，造成 UnboundLocalError

# 解决方案：
# 1. 修复 _generate_phenotype_stats 方法的缩进
# 2. 增加异常处理以防止类似问题
# 3. 改进数据验证

# ============================================================================
# 请将以下代码替换 colony_analysis/utils/results.py 中对应的方法
# ============================================================================






# ============================================================================
# 使用说明：
# 1. 备份原始的 colony_analysis/utils/results.py 文件
# 2. 用以上修复的方法替换原文件中对应的方法
# 3. 重新运行程序
# ============================================================================
