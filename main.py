
#!/usr/bin/env python3
"""
Colony Detection SAM 3.0 - 完整重构版本
基于SAM的链霉菌菌落检测和分析工具
"""

# ============================================================================
# 1. main.py - 主入口文件（简洁版）
# ============================================================================

import argparse
from colony_analysis.config.settings import ConfigManager
import logging
import sys
import time
import os
from pathlib import Path
from tqdm import tqdm

from colony_analysis.pipeline import AnalysisPipeline, batch_medium_pipeline
from colony_analysis.pairing import pair_colonies_across_views
from colony_analysis.utils.file_utils import collect_all_images, parse_filename
def get_output_path(image_name: str, replicate: str, orientation: str, output_root: str) -> str:
    """
    从图像文件名中提取组名、培养基、日期编号等信息，并构造输出路径。
    示例输入: image_name='Lib96_Ctrl_@MMM_Front20250401_09202932'
    replicate='01', orientation='Front'
    返回:
    'output/Lib96_Ctrl/MMM/20250401_09202932/replicate_01/Front/'
    """
    # 分割组名与其他信息
    parts = image_name.split('_@')
    if len(parts) != 2:
        raise ValueError(f"图像名格式错误: {image_name}")
    group_name = parts[0]                      # e.g., 'Lib96_Ctrl'
    rest = parts[1]                            # e.g., 'MMM_Front20250401_09202932'
    # 找到medium
    medium = rest.split('Front')[0].split('Back')[0].rstrip('_')
    # 找到full_id
    full_id = ''
    if 'Front' in rest:
        full_id = rest.split('Front')[1]
        orientation_str = 'Front'
    elif 'Back' in rest:
        full_id = rest.split('Back')[1]
        orientation_str = 'Back'
    else:
        raise ValueError(f"图像名未包含Front或Back: {image_name}")
    # 去除前导下划线
    full_id = full_id.lstrip('_')
    # 规范化orientation
    orientation_str = orientation.capitalize()
    return os.path.join(output_root, group_name, medium, full_id, f"replicate_{replicate}", orientation_str)


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

  批处理模式:
    python main.py -I Image_input -o results/              # 处理整个目录
    python main.py -I Image_input -o results/ --filter-medium MM --filter-side back  # 只处理特定条件
    
  单图像模式:
    python main.py -i image.jpg -o results/ --medium MM --side back
        """,
    )

    # 输入/输出
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i","--input", nargs="+", help="输入图像路径，可指定多个")
    group.add_argument("--input-dir", "-I","--dir", help="包含待分析图像的目录")
    parser.add_argument(
        "--output", "-o", default="output", help="输出目录 (默认: output)"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="在批处理模式下与用户交互确认"
    )

    # 批处理模式的过滤选项
    parser.add_argument("--filter-medium", choices=["r5", "mm", "mmm", "all"], default="all",
                       help="批处理时只处理特定培养基 (默认: all)")
    parser.add_argument("--filter-side", choices=["front", "back", "all"], default="all", 
                       help="批处理时只处理特定拍摄面 (默认: all)")
    parser.add_argument("--filter-sample", help="批处理时只处理包含特定字符串的样本名")
    
    

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

    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )


    # 孔板参数
    parser.add_argument("--well-plate", action="store_true", help="使用96孔板编号系统")
    parser.add_argument("--rows", type=int, default=8, help="孔板行数 (默认: 8)")
    parser.add_argument("--cols", type=int, default=12, help="孔板列数 (默认: 12)")
    parser.add_argument("--force-96plate-detection", action="store_true", help="是否强制使用96孔板布局进行检测")
    parser.add_argument("--fallback-null-policy", type=str, default="fill", choices=["fill", "null", "skip"], help="未检测到菌落时的处理策略")
    parser.add_argument("--outlier-detection", action="store_true", help="是否启用离群值检测")
    parser.add_argument("--outlier-metric", type=str, default="area", help="离群值检测使用的指标 (如 area/intensity/shape_metric)")
    parser.add_argument("--outlier-threshold", type=float, default=3.0, help="离群值检测的Z-score阈值")

    # 配置参数
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument(
        "--min-area", type=int, default=2000, help="最小菌落面积 (默认: 2000)"
    )
    parser.add_argument("--medium", help="培养基名称 (单图像模式必需)")
    parser.add_argument("--side", choices=["front", "back"], help="图像正面或背面 (单图像模式必需)")
    parser.add_argument("--device", choices=["cpu","cuda"], default="cuda", help="运行设备")

    # 解析参数
    parser.add_argument(
        "--use_cp_sam_masks",
        action="store_true",
        help="Use precomputed Cellpose-SAM masks (*.tif) instead of running SAMModel"
    )
    parser.add_argument(
        "--cp_sam_masks_dir",
        type=str,
        default=None,
        help="Directory containing your *_cp_masks.tif mask files"
    )

    args = parser.parse_args()
    # Require medium and side only in single-image mode
    if args.image and (args.medium is None or args.side is None):
        parser.error("--medium and --side are required when using --image")
    return args


def setup_logging(verbose: bool = False, log_level: str = "INFO") -> None:
    """配置日志系统并与 ``tqdm`` 进度条兼容"""

    class TqdmHandler(logging.StreamHandler):
        """自定义处理器，使用 ``tqdm.write`` 输出日志，避免打断进度条"""

        def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console_handler = TqdmHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(console_handler)

    # 创建文件日志处理器
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"colony_analysis_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)

    logging.info(f"日志记录到文件: {log_file}")

def log_command_line_info(args):
    """记录命令行信息到日志"""
    
    # 1. 记录原始命令行
    original_command = ' '.join(sys.argv)
    logging.info("=" * 80)
    logging.info("🚀 Colony Detection SAM 2.0 启动")
    logging.info("=" * 80)
    logging.info(f"📝 执行命令: {original_command}")
    
    # 2. 记录解析后的参数（格式化显示）
    logging.info("📋 解析后的参数:")
    args_dict = vars(args)
    
    # 按类别分组显示参数
    param_groups = {
        "输入/输出": ["image", "input_dir", "output"],
        "检测参数": ["mode", "model", "device", "min_area"],
        "96孔板参数": ["well_plate", "rows", "cols", "force_96plate_detection", "fallback_null_policy"],
        "分析参数": ["advanced", "debug", "verbose", "outlier_detection"],
        "过滤参数": ["filter_medium", "filter_side", "filter_sample"],
        "培养基参数": ["medium", "side"],
        "配置参数": ["config", "log_level"]
    }
    
    for group_name, param_names in param_groups.items():
        group_params = {k: v for k, v in args_dict.items() if k in param_names and v is not None}
        if group_params:  # 只显示有值的参数组
            logging.info(f"  {group_name}:")
            for key, value in group_params.items():
                logging.info(f"    {key}: {value}")
    
    # 3. 记录其他重要信息
    logging.info("🖥️  系统信息:")
    logging.info(f"    Python版本: {sys.version}")
    logging.info(f"    工作目录: {os.getcwd()}")
    logging.info(f"    启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 4. 检查关键文件/目录
    logging.info("📁 文件检查:")
    if hasattr(args, 'image') and args.image:
        for img_path in (args.image if isinstance(args.image, list) else [args.image]):
            exists = "✅" if Path(img_path).exists() else "❌"
            logging.info(f"    输入图像: {exists} {img_path}")
    
    if hasattr(args, 'input_dir') and args.input_dir:
        exists = "✅" if Path(args.input_dir).exists() else "❌"
        logging.info(f"    输入目录: {exists} {args.input_dir}")
    
    if hasattr(args, 'config') and args.config:
        exists = "✅" if Path(args.config).exists() else "❌"
        logging.info(f"    配置文件: {exists} {args.config}")
    
    # 检查模型文件
    model_file = f"models/sam_{args.model}_*.pth"
    model_exists = len(list(Path("models").glob(f"sam_{args.model}_*.pth"))) > 0 if Path("models").exists() else False
    status = "✅" if model_exists else "❌"
    logging.info(f"    SAM模型: {status} {model_file}")
    
    logging.info("=" * 80)

def save_command_to_history(args, output_dir):
    """将命令行参数保存到历史文件"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建命令历史记录
        history_file = output_path / "command_history.json"
        
        command_record = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "original_command": ' '.join(sys.argv),
            "parsed_args": vars(args),
            "working_directory": os.getcwd(),
            "python_version": sys.version,
        }
        
        # 如果历史文件存在，追加；否则创建新文件
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    history = [history]  # 兼容旧格式
            except:
                history = []
        else:
            history = []
        
        history.append(command_record)
        
        # 保存历史记录（最多保留最近50条）
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history[-50:], f, indent=2, ensure_ascii=False)
        
        logging.info(f"📝 命令记录已保存到: {history_file}")
        
    except Exception as e:
        logging.warning(f"保存命令历史失败: {e}")



def main():
    """主函数 - 保持简洁，主要负责流程协调"""
    try:
        # 解析参数
        args = parse_arguments()

        # 设置日志
        setup_logging(args.verbose)
        # 🔥 新增：记录命令行信息
        log_command_line_info(args)
        
        # 🔥 新增：保存命令到历史文件
        if hasattr(args, 'output'):
            save_command_to_history(args, args.output)
        # 显示启动信息
        print_startup_banner()

        # ========== 批处理模式 ==========
        if args.input_dir:
            print(f"\n📁 批处理模式")
            print(f"   输入目录: {args.input_dir}")
            print(f"   输出目录: {args.output}")
            
            # 收集图像文件
            imgs = collect_all_images(args.input_dir)
            if not imgs:
                raise FileNotFoundError("在指定目录中未找到图像文件")
            
            # 应用过滤（如果提供了过滤参数）
            filtered_imgs = imgs
            if hasattr(args, 'filter_medium') and args.filter_medium != 'all':
                filtered_imgs = [img for img in filtered_imgs 
                               if args.filter_medium.lower() in img.stem.lower()]
                print(f"   培养基过滤: {args.filter_medium} ({len(filtered_imgs)}/{len(imgs)} 个文件)")
            
            if hasattr(args, 'filter_side') and args.filter_side != 'all':
                filtered_imgs = [img for img in filtered_imgs 
                               if args.filter_side.lower() in img.stem.lower()]
                print(f"   拍摄面过滤: {args.filter_side} ({len(filtered_imgs)}/{len(imgs)} 个文件)")
            
            if hasattr(args, 'filter_sample') and args.filter_sample:
                filtered_imgs = [img for img in filtered_imgs 
                               if args.filter_sample.lower() in img.stem.lower()]
                print(f"   样本名过滤: {args.filter_sample} ({len(filtered_imgs)}/{len(imgs)} 个文件)")
            
            if not filtered_imgs:
                print("❌ 没有符合过滤条件的图像文件")
                return 1
            
            # 交互确认
            if args.interactive:
                print(f"\n即将处理 {len(filtered_imgs)} 个图像文件:")
                # 显示前5个和后2个文件
                if len(filtered_imgs) <= 7:
                    for img in filtered_imgs:
                        print(f"  - {img.name}")
                else:
                    for img in filtered_imgs[:5]:
                        print(f"  - {img.name}")
                    print(f"  ... (还有 {len(filtered_imgs)-7} 个文件)")
                    for img in filtered_imgs[-2:]:
                        print(f"  - {img.name}")
                
                cont = input("\n继续? [y/N]: ").strip().lower()
                if cont != "y":
                    return 0
            
            # 执行批处理
            # 传递额外的参数给批处理函数
            batch_args = {
                'device': args.device,
                'mode': args.mode,
                'model': args.model,
                'min_area': args.min_area,
                'force_96plate_detection': args.force_96plate_detection,
                'fallback_null_policy': args.fallback_null_policy,
                'debug': args.debug,
                'well_plate': args.well_plate,
                'rows': args.rows,
                'cols': args.cols,
            }
            
            # 使用过滤后的图像列表
            batch_medium_pipeline(
                filtered_imgs,  # 传递过滤后的图像列表而不是目录
                args.output,
                **batch_args
            )
            
            # 配对前后视图
            pair_colonies_across_views(args.output)
            
            print(f"\n✅ 批处理完成！")
            print(f"   处理图像: {len(filtered_imgs)} 个")
            print(f"   输出目录: {args.output}")
            
            return 0

        # ========== 单图像/多图像模式 ==========
        else:
            # 单图像模式必须有 medium 和 side 参数
            if not args.medium or not args.side:
                print("\n❌ 错误：单图像模式需要指定 --medium 和 --side 参数")
                print("\n示例:")
                print("  python main.py -i image.jpg -o results --medium MM --side back")
                print("\n或使用批处理模式:")
                print("  python main.py -I Image_input -o results")
                return 1
            
            images = args.image

            if args.interactive and len(images) > 1:
                print("即将处理以下图像:\n" + "\n".join(images))
                cont = input("继续? [y/N]: ").strip().lower()
                if cont != "y":
                    return 0

            # load and merge config into a ConfigManager instance
            config_manager = ConfigManager(args.config)
            config_manager.update_from_args(args)
            cfg = config_manager
            
            # 处理每个图像
            for i, img in enumerate(images):
                print(f"\n处理图像 {i+1}/{len(images)}: {Path(img).name}")
                
                img_output = Path(args.output)
                sample_name = replicate = None
                stem = Path(img).stem
                
                # 根据文件名推断输出结构
                try:
                    sample_name, _, _, replicate = parse_filename(stem)
                except Exception:
                    pass

                orientation = args.side
                medium = args.medium
                
                if sample_name and replicate:
                    output_path = get_output_path(stem, replicate, orientation, args.output)
                    os.makedirs(output_path, exist_ok=True)
                    img_output = Path(output_path)
                elif len(images) > 1:
                    img_output = img_output / stem
                    os.makedirs(img_output, exist_ok=True)
                else:
                    os.makedirs(img_output, exist_ok=True)

                img_args = argparse.Namespace(**vars(args))
                img_args.image = img
                img_args.output = str(img_output)
                img_args.medium = medium
                img_args.orientation = orientation
                img_args.replicate = replicate
                img_args.cfg = cfg
                
                pipeline = AnalysisPipeline(img_args)
                results = pipeline.run()
                print_completion_summary(results)
                pair_colonies_across_views(results["output_dir"])

            return 0

    except KeyboardInterrupt:
        logging.info("\n用户中断程序执行")
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
        print("\n✅ 分析完成!")
        print(f"   检测菌落: {results.get('total_colonies', 0)} 个")
        print(f"   处理时间: {results.get('elapsed_time', 0):.2f} 秒")
        print(f"   输出目录: {results.get('output_dir', 'N/A')}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
