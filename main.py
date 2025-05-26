#!/usr/bin/env python3
"""
Colony Detection SAM 2.0 - 完整重构版本
基于SAM的链霉菌菌落检测和分析工具
"""

# ============================================================================
# 1. main.py - 主入口文件（简洁版）
# ============================================================================

import os
import sys
import time
import argparse
import logging
from pathlib import Path


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Colony Detection SAM 2.0 - 基于SAM的链霉菌菌落分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py -i image.jpg -o results/                    # 基本分析
  python main.py -i image.jpg -o results/ --advanced --debug # 高级分析
  python main.py -i plate.jpg --well-plate --mode grid       # 96孔板模式
        """
    )

    # 必需参数
    parser.add_argument('--image', '-i', required=True,
                        help='输入图像路径')
    parser.add_argument('--output', '-o', default='output',
                        help='输出目录 (默认: output)')

    # 检测参数
    parser.add_argument('--mode', '-m',
                        choices=['auto', 'grid', 'hybrid'],
                        default='auto',
                        help='检测模式 (默认: auto)')
    parser.add_argument('--model',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        default='vit_b',
                        help='SAM模型类型 (默认: vit_b)')

    # 分析参数
    parser.add_argument('--advanced', '-a', action='store_true',
                        help='启用高级特征分析')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='启用调试模式，生成详细输出')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='输出详细日志信息')

    # 孔板参数
    parser.add_argument('--well-plate', action='store_true',
                        help='使用96孔板编号系统')
    parser.add_argument('--rows', type=int, default=8,
                        help='孔板行数 (默认: 8)')
    parser.add_argument('--cols', type=int, default=12,
                        help='孔板列数 (默认: 12)')

    # 配置参数
    parser.add_argument('--config', type=str,
                        help='配置文件路径')
    parser.add_argument('--min-area', type=int, default=5000,
                        help='最小菌落面积 (默认: 5000)')

    return parser.parse_args()


def setup_logging(verbose=False):
    """配置日志系统"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

    # 创建文件日志处理器
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'colony_analysis_{timestamp}.log'

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(format_str))
    logging.getLogger().addHandler(file_handler)

    logging.info(f"日志记录到文件: {log_file}")


def main():
    """主函数 - 保持简洁，主要负责流程协调"""
    try:
        # 解析参数
        args = parse_arguments()

        # 设置日志
        setup_logging(args.verbose)

        # 显示启动信息
        print_startup_banner()

        # 执行分析流程
        from colony_analysis.pipeline import AnalysisPipeline

        pipeline = AnalysisPipeline(args)
        results = pipeline.run()

        # 显示完成信息
        print_completion_summary(results)

        return 0

    except KeyboardInterrupt:
        logging.info("用户中断程序执行")
        return 1
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            logging.error(traceback.format_exc())
        return 1


def print_startup_banner():
    """显示启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        Colony Detection SAM 2.0                             ║
║                     基于SAM的链霉菌菌落分析工具                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_completion_summary(results):
    """显示完成摘要"""
    if results:
        print(f"\n✅ 分析完成!")
        print(f"   检测菌落: {results.get('total_colonies', 0)} 个")
        print(f"   处理时间: {results.get('elapsed_time', 0):.2f} 秒")
        print(f"   输出目录: {results.get('output_dir', 'N/A')}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
