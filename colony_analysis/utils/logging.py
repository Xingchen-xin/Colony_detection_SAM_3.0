# 日志管理代码将放置在这里# colony_analysis/utils/logging.py
import logging
import os
import time
from pathlib import Path
import platform
import sys


class LogManager:
    """日志管理器"""

    def __init__(self, config=None):
        """
        初始化日志管理器
        
        Args:
            config: 配置管理器
        """
        # 设置日志级别
        log_level = 'INFO'
        log_to_file = True
        log_dir = None

        # 从配置获取日志设置
        if config is not None:
            if hasattr(config, 'get'):
                log_level = config.get('logging', 'level', 'INFO')
                log_to_file = config.get('logging', 'log_to_file', True)
                log_dir = config.get('logging', 'log_dir', None)

        # 设置日志级别
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO

        # 配置日志格式
        console_format = '%(levelname)s: %(message)s'
        file_format = '%(asctime)s - %(levelname)s - %(message)s'

        # 配置根日志
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # 清除现有处理程序
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 添加控制台处理程序
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(console_format))
        root_logger.addHandler(console_handler)

        # 添加文件处理程序(如果启用)
        if log_to_file:
            if log_dir is None:
                log_dir = Path.home() / '.colony_analysis' / 'logs'

            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)

            # 创建日志文件名
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(
                log_dir, f'colony_analysis_{timestamp}.log')

            # 添加文件处理程序
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(file_format))
            root_logger.addHandler(file_handler)

            logging.info(f"日志记录到文件: {log_file}")

    def log_system_info(self):
        """记录系统信息"""
        logging.info("=" * 50)
        logging.info("系统信息:")
        logging.info(f"操作系统: {platform.platform()}")
        logging.info(f"Python版本: {platform.python_version()}")
        logging.info(f"解释器路径: {sys.executable}")

        # 记录CUDA信息
        try:
            import torch
            if torch.cuda.is_available():
                logging.info(f"CUDA可用: 是")
                logging.info(f"CUDA版本: {torch.version.cuda}")
                logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                logging.info("CUDA可用: 否")
        except ImportError:
            logging.info("CUDA状态: 未安装PyTorch")

        logging.info("=" * 50)
