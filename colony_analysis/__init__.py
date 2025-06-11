# ============================================================================
# 2. colony_analysis/__init__.py - 包初始化
# ============================================================================

"""
Colony Analysis Package 2.0
基于SAM的链霉菌菌落检测和分析工具

版本: 2.0.0
作者: Colony Analysis Team
"""

__version__ = "2.0.0"
__author__ = "Colony Analysis Team"

# 延迟导入主要模块，避免在导入包时触发繁重依赖
def __getattr__(name):
    if name == "ColonyAnalyzer":
        from .analysis import ColonyAnalyzer
        return ColonyAnalyzer
    if name == "FeatureExtractor":
        from .analysis import FeatureExtractor
        return FeatureExtractor
    if name == "ScoringSystem":
        from .analysis import ScoringSystem
        return ScoringSystem
    if name == "ConfigManager":
        from .config import ConfigManager
        return ConfigManager
    if name == "SAMModel":
        from .core import SAMModel
        return SAMModel
    if name == "ColonyDetector":
        from .core import ColonyDetector
        return ColonyDetector
    if name == "LogManager":
        from .utils import LogManager
        return LogManager
    if name == "ResultManager":
        from .utils import ResultManager
        return ResultManager
    if name == "Visualizer":
        from .utils import Visualizer
        return Visualizer
    raise AttributeError(name)

__all__ = [
    "ConfigManager",
    "SAMModel",
    "ColonyDetector",
    "ColonyAnalyzer",
    "FeatureExtractor",
    "ScoringSystem",
    "LogManager",
    "ResultManager",
    "Visualizer",
]
