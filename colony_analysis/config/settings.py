# ============================================================================
# 4. colony_analysis/config/settings.py - 配置管理
# ============================================================================

import os
import json
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class DetectionConfig:
    """检测配置"""
    model_type: str = 'vit_b'
    mode: str = 'auto'
    min_colony_area: int = 5000
    expand_pixels: int = 8
    merge_overlapping: bool = True
    use_preprocessing: bool = True
    overlap_threshold: float = 0.3


@dataclass
class SAMConfig:
    """SAM模型配置"""
    points_per_side: int = 64
    pred_iou_thresh: float = 0.85
    stability_score_thresh: float = 0.8
    min_mask_region_area: int = 1500
    crop_n_layers: int = 1
    crop_n_points_downscale_factor: int = 1


@dataclass
class AnalysisConfig:
    """分析配置"""
    advanced: bool = False
    learning_enabled: bool = False
    aerial_threshold: float = 0.6
    metabolite_threshold: float = 0.5
    enable_parallel: bool = False
    max_workers: int = 4


@dataclass
class OutputConfig:
    """输出配置"""
    debug: bool = False
    well_plate: bool = False
    rows: int = 8
    cols: int = 12
    save_masks: bool = True
    save_visualizations: bool = True
    image_format: str = 'jpg'


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = 'INFO'
    log_to_file: bool = True
    log_dir: Optional[str] = None
    max_log_files: int = 10


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器"""
        self.config_path = self._resolve_config_path(config_path)

        # 初始化配置对象
        self.detection = DetectionConfig()
        self.sam = SAMConfig()
        self.analysis = AnalysisConfig()
        self.output = OutputConfig()
        self.logging = LoggingConfig()

        # 加载配置
        self._load_config()

    def _resolve_config_path(self, config_path: Optional[str]) -> str:
        """解析配置文件路径"""
        if config_path and Path(config_path).exists():
            return config_path

        # 默认配置文件位置
        default_locations = [
            'config.yaml',
            Path.home() / '.colony_analysis' / 'config.yaml',
            Path(__file__).parent.parent.parent / 'config.yaml'
        ]

        for path in default_locations:
            if Path(path).exists():
                return str(path)

        # 返回默认路径（可能不存在）
        return str(Path.home() / '.colony_analysis' / 'config.yaml')

    def _load_config(self):
        """从文件加载配置"""
        if not Path(self.config_path).exists():
            logging.info(f"配置文件不存在，使用默认配置: {self.config_path}")
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.json'):
                    config_data = json.load(f)
                else:
                    config_data = yaml.safe_load(f) or {}

            self._update_config_from_dict(config_data)
            logging.info(f"已从 {self.config_path} 加载配置")

        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")

    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """从字典更新配置"""
        for section_name, section_data in config_data.items():
            if not isinstance(section_data, dict) or not hasattr(self, section_name):
                continue

            config_obj = getattr(self, section_name)
            for field_name, field_value in section_data.items():
                if hasattr(config_obj, field_name):
                    setattr(config_obj, field_name, field_value)

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """获取配置值"""
        if not hasattr(self, section):
            return default

        config_obj = getattr(self, section)
        if key is None:
            return config_obj

        return getattr(config_obj, key, default)

    def update_from_args(self, args):
        """从命令行参数更新配置"""
        # 检测配置
        if hasattr(args, 'model') and args.model:
            self.detection.model_type = args.model
        if hasattr(args, 'mode') and args.mode:
            self.detection.mode = args.mode
        if hasattr(args, 'min_area') and args.min_area:
            self.detection.min_colony_area = args.min_area

        # 分析配置
        if hasattr(args, 'advanced') and args.advanced:
            self.analysis.advanced = True

        # 输出配置
        if hasattr(args, 'debug') and args.debug:
            self.output.debug = True
        if hasattr(args, 'well_plate') and args.well_plate:
            self.output.well_plate = True
        if hasattr(args, 'rows') and args.rows:
            self.output.rows = args.rows
        if hasattr(args, 'cols') and args.cols:
            self.output.cols = args.cols

        # 日志配置
        if hasattr(args, 'verbose') and args.verbose:
            self.logging.level = 'DEBUG'
