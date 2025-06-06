#!/usr/bin/env python3
"""
Colony Detection SAM 2.0 - 完整重构版本
基于SAM的链霉菌菌落检测和分析工具
"""

# ============================================================================
# 1. main.py - 主入口文件（简洁版）
# ============================================================================

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from colony_analysis.pipeline import AnalysisPipeline, batch_medium_pipeline
from colony_analysis.utils.file_utils import collect_all_images, parse_filename


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
        """,
    )

    # 输入/输出
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", nargs="+", help="输入图像路径，可指定多个")
    group.add_argument("--input-dir", "-I", help="包含待分析图像的目录")
    parser.add_argument(
        "--output", "-o", default="output", help="输出目录 (默认: output)"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="在批处理模式下与用户交互确认"
    )

    # 检测参数
    parser.add_argument(
        "--mode",
        "-m",
        choices=["auto", "grid", "hybrid"],
        default="auto",
        help="检测模式 (默认: auto)",
    )
    parser.add_argument(
        "--model",
        choices=["vit_b", "vit_l", "vit_h"],
        default="vit_b",
        help="SAM模型类型 (默认: vit_b)",
    )

    # 分析参数
    parser.add_argument(
        "--advanced", "-a", action="store_true", help="启用高级特征分析"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="启用调试模式，生成详细输出"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="输出详细日志信息")

    # 孔板参数
    parser.add_argument("--well-plate", action="store_true", help="使用96孔板编号系统")
    parser.add_argument("--rows", type=int, default=8, help="孔板行数 (默认: 8)")
    parser.add_argument("--cols", type=int, default=12, help="孔板列数 (默认: 12)")

    # 配置参数
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument(
        "--min-area", type=int, default=2000, help="最小菌落面积 (默认: 2000)"
    )

    return parser.parse_args()


def setup_logging(verbose=False):
    """配置日志系统"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # 创建文件日志处理器
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"colony_analysis_{timestamp}.log"

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

        images = []
        if args.input_dir:
            if args.interactive:
                imgs = collect_all_images(args.input_dir)
                if not imgs:
                    raise FileNotFoundError("在指定目录中未找到图像文件")
                print("即将处理以下图像:\n" + "\n".join(str(p) for p in imgs))
                cont = input("继续? [y/N]: ").strip().lower()
                if cont != "y":
                    return 0
            batch_medium_pipeline(args.input_dir, args.output)
            return 0
        else:
            images = args.image

        if args.interactive and len(images) > 1:
            print("即将处理以下图像:\n" + "\n".join(images))
            cont = input("继续? [y/N]: ").strip().lower()
            if cont != "y":
                return 0

        for img in images:
            img_output = Path(args.output)
            sample_name = medium = orientation = replicate = None
            stem = Path(img).stem
            # 根据文件名推断输出结构
            try:
                sample_name, medium, orientation, replicate = parse_filename(stem)
            except Exception:
                pass

            if sample_name and medium and orientation:
                img_output = (
                    img_output
                    / sample_name
                    / medium.upper()
                    / orientation.capitalize()
                    / f"replicate_{replicate}"
                )
            elif len(images) > 1:
                img_output = img_output / stem

            img_args = argparse.Namespace(**vars(args))
            img_args.image = img
            img_args.output = str(img_output)
            img_args.medium = medium
            img_args.orientation = orientation
            img_args.replicate = replicate
            pipeline = AnalysisPipeline(img_args)
            results = pipeline.run()
            print_completion_summary(results)

        return 0

    except KeyboardInterrupt:
        logging.info("用户中断程序执行")
        return 1
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        if hasattr(args, "debug") and args.debug:
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
