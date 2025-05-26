# ============================================================================
# 20. 使用示例 - examples/basic_usage.py
# ============================================================================

BASIC_USAGE_PY = """#!/usr/bin/env python3
\"\"\"
Colony Detection SAM 2.0 基本使用示例
\"\"\"

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from colony_analysis.config import ConfigManager
from colony_analysis.core import SAMModel, ColonyDetector
from colony_analysis.analysis import ColonyAnalyzer
from colony_analysis.utils import ResultManager
import cv2


def basic_analysis_example():
    \"\"\"基本分析示例\"\"\"
    
    print(\"🔬 Colony Detection SAM 2.0 基本使用示例\")
    print(\"=\" * 50)
    
    # 1. 初始化配置
    config = ConfigManager()
    print(\"✅ 配置管理器初始化完成\")
    
    # 2. 加载测试图像（这里需要替换为实际图像路径）
    test_image_path = \"test_image.jpg\"  # 替换为实际图像路径
    
    if not os.path.exists(test_image_path):
        print(f\"❌ 测试图像不存在: {test_image_path}\")
        print(\"请将测试图像放在项目根目录下，并命名为 test_image.jpg\")
        return
    
    # 加载图像
    img = cv2.imread(test_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f\"✅ 图像加载完成，尺寸: {img_rgb.shape}\")
    
    try:
        # 3. 初始化SAM模型
        print(\"📥 初始化SAM模型...\")
        sam_model = SAMModel(model_type='vit_b', config=config)
        print(\"✅ SAM模型初始化完成\")
        
        # 4. 初始化检测器
        detector = ColonyDetector(sam_model=sam_model, config=config)
        print(\"✅ 检测器初始化完成\")
        
        # 5. 执行检测
        print(\"🔍 开始菌落检测...\")
        colonies = detector.detect(img_rgb, mode='auto')
        print(f\"✅ 检测完成，发现 {len(colonies)} 个菌落\")
        
        # 6. 初始化分析器
        analyzer = ColonyAnalyzer(sam_model=sam_model, config=config)
        
        # 7. 执行分析
        print(\"📊 开始菌落分析...\")
        analyzed_colonies = analyzer.analyze(colonies, advanced=False)
        print(f\"✅ 分析完成，共 {len(analyzed_colonies)} 个菌落\")
        
        # 8. 保存结果
        print(\"💾 保存结果...\")
        result_manager = ResultManager('example_output')
        
        # 模拟args对象
        class Args:
            def __init__(self):
                self.advanced = False
                self.debug = True
                self.mode = 'auto'
                self.model = 'vit_b'
        
        args = Args()
        result_manager.save_all_results(analyzed_colonies, args)
        print(\"✅ 结果保存完成，输出目录: example_output/\")
        
        # 9. 显示基本统计
        print(\"\\n📈 分析结果摘要:\")
        print(f\"总菌落数: {len(analyzed_colonies)}\")
        
        # 计算平均面积
        areas = [colony.get('area', 0) for colony in analyzed_colonies]
        if areas:
            avg_area = sum(areas) / len(areas)
            print(f\"平均面积: {avg_area:.2f} 像素\")
        
        # 统计表型分布
        phenotypes = {}
        for colony in analyzed_colonies:
            dev_state = colony.get('phenotype', {}).get('development_state', 'unknown')
            phenotypes[dev_state] = phenotypes.get(dev_state, 0) + 1
        
        print(\"发育状态分布:\")
        for phenotype, count in phenotypes.items():
            print(f\"  {phenotype}: {count}\")
        
        print(\"\\n🎉 示例运行完成！\")
        
    except FileNotFoundError as e:
        print(f\"❌ 模型文件未找到: {e}\")
        print(\"请确保已下载SAM模型权重文件到 models/ 目录\")
        
    except Exception as e:
        print(f\"❌ 运行时错误: {e}\")
        import traceback
        traceback.print_exc()


def advanced_analysis_example():
    \"\"\"高级分析示例\"\"\"
    
    print(\"\\n🔬 高级分析示例\")
    print(\"=\" * 30)
    
    # 这里可以添加更复杂的分析流程
    print(\"高级分析功能包括:\")
    print(\"- 扩散区域检测\")
    print(\"- 详细特征提取\")
    print(\"- 可视化生成\")
    print(\"- 详细报告\")


if __name__ == \"__main__\":
    # 运行基本示例
    basic_analysis_example()
    
    # 运行高级示例
    advanced_analysis_example()
"""

