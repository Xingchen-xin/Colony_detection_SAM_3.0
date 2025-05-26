# ============================================================================
# 3. colony_analysis/pipeline.py - 分析管道
# ============================================================================

import time
import logging
import cv2
from pathlib import Path

from .config import ConfigManager
from .core import SAMModel, ColonyDetector
from .analysis import ColonyAnalyzer
from .utils import ResultManager, Visualizer, ImageValidator


class AnalysisPipeline:
    """分析管道 - 协调整个分析流程"""

    def __init__(self, args):
        """初始化分析管道"""
        self.args = args
        self.start_time = None
        self.config = None
        self.sam_model = None
        self.detector = None
        self.analyzer = None
        self.result_manager = None

    def run(self):
        """运行完整的分析流程"""
        self.start_time = time.time()

        try:
            # 1. 初始化组件
            self._initialize_components()

            # 2. 加载和验证图像
            img_rgb = self._load_and_validate_image()

            # 3. 执行检测
            colonies = self._detect_colonies(img_rgb)

            # 4. 执行分析
            analyzed_colonies = self._analyze_colonies(colonies)

            # 5. 保存结果
            self._save_results(analyzed_colonies, img_rgb)

            # 6. 返回结果摘要
            return self._generate_summary(analyzed_colonies)

        except Exception as e:
            logging.error(f"分析管道执行失败: {e}")
            raise

    def _initialize_components(self):
        """初始化所有组件"""
        logging.info("初始化组件...")

        # 配置管理器
        self.config = ConfigManager(self.args.config)
        self.config.update_from_args(self.args)

        # SAM模型
        self.sam_model = SAMModel(
            model_type=self.args.model,
            config=self.config
        )

        # 检测器
        self.detector = ColonyDetector(
            sam_model=self.sam_model,
            config=self.config
        )

        # 分析器
        self.analyzer = ColonyAnalyzer(
            sam_model=self.sam_model,
            config=self.config
        )

        # 结果管理器
        self.result_manager = ResultManager(self.args.output)

        logging.info("组件初始化完成")

    def _load_and_validate_image(self):
        """加载和验证图像"""
        logging.info(f"加载图像: {self.args.image}")

        # 检查文件是否存在
        if not Path(self.args.image).exists():
            raise FileNotFoundError(f"图像文件不存在: {self.args.image}")

        # 加载图像
        img = cv2.imread(self.args.image)
        if img is None:
            raise ValueError(f"无法读取图像文件: {self.args.image}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 验证图像
        is_valid, error_msg = ImageValidator.validate_image(img_rgb)
        if not is_valid:
            raise ValueError(f"图像验证失败: {error_msg}")

        logging.info(f"图像加载成功，尺寸: {img_rgb.shape}")
        return img_rgb

    def _detect_colonies(self, img_rgb):
        """执行菌落检测"""
        logging.info("开始菌落检测...")

        colonies = self.detector.detect(
            img_rgb,
            mode=self.args.mode
        )

        if not colonies:
            raise ValueError("未检测到任何菌落，请检查图像或调整参数")

        logging.info(f"检测到 {len(colonies)} 个菌落")
        return colonies

    def _analyze_colonies(self, colonies):
        """执行菌落分析"""
        logging.info("开始菌落分析...")

        analyzed_colonies = self.analyzer.analyze(
            colonies,
            advanced=self.args.advanced
        )

        logging.info(f"分析完成，共 {len(analyzed_colonies)} 个菌落")
        return analyzed_colonies

    def _save_results(self, analyzed_colonies, img_rgb):
        """保存结果"""
        logging.info("保存分析结果...")

        # 保存基本结果
        self.result_manager.save_all_results(analyzed_colonies, self.args)

        # 生成可视化
        if self.args.debug:
            visualizer = Visualizer(self.args.output)
            visualizer.create_debug_visualizations(img_rgb, analyzed_colonies)

        logging.info(f"结果已保存到: {self.args.output}")

    def _generate_summary(self, analyzed_colonies):
        """生成结果摘要"""
        elapsed_time = time.time() - self.start_time

        return {
            'total_colonies': len(analyzed_colonies),
            'elapsed_time': elapsed_time,
            'output_dir': self.args.output,
            'mode': self.args.mode,
            'model': self.args.model,
            'advanced': self.args.advanced
        }
