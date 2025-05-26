# ============================================================================
# 18. 项目初始化脚本 - setup_project.py
# ============================================================================

SETUP_PROJECT_PY = """#!/usr/bin/env python3
\"\"\"
Colony Detection SAM 2.0 项目初始化脚本
运行此脚本将创建完整的项目目录结构
\"\"\"

import os
import sys
from pathlib import Path


def create_project_structure():
    \"\"\"创建项目目录结构\"\"\"
    
    # 项目根目录
    project_root = Path.cwd()
    
    # 创建主要目录
    directories = [
        'colony_analysis/config',
        'colony_analysis/core', 
        'colony_analysis/analysis',
        'colony_analysis/utils',
        'tests',
        'models',
        'examples/images',
        'examples/notebooks',
        'docs',
        'logs'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
    
    # 创建 __init__.py 文件
    init_files = [
        'colony_analysis/__init__.py',
        'colony_analysis/config/__init__.py',
        'colony_analysis/core/__init__.py',
        'colony_analysis/analysis/__init__.py',
        'colony_analysis/utils/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_file in init_files:
        init_path = project_root / init_file
        if not init_path.exists():
            init_path.write_text('# 包初始化文件\\n')
            print(f"📄 创建文件: {init_path}")
    
    # 创建 .gitkeep 文件
    gitkeep_files = [
        'models/.gitkeep',
        'examples/images/.gitkeep', 
        'examples/notebooks/.gitkeep',
        'docs/.gitkeep'
    ]
    
    for gitkeep_file in gitkeep_files:
        gitkeep_path = project_root / gitkeep_file
        if not gitkeep_path.exists():
            gitkeep_path.write_text('# 目录占位文件\\n')
            print(f"📄 创建文件: {gitkeep_path}")


def create_config_files():
    \"\"\"创建配置文件\"\"\"
    
    project_root = Path.cwd()
    
    # config.yaml
    if not (project_root / 'config.yaml').exists():
        (project_root / 'config.yaml').write_text(CONFIG_YAML)
        print(f"📄 创建配置文件: config.yaml")
    
    # requirements.txt
    if not (project_root / 'requirements.txt').exists():
        (project_root / 'requirements.txt').write_text(REQUIREMENTS_TXT)
        print(f"📄 创建依赖文件: requirements.txt")
    
    # README.md
    if not (project_root / 'README.md').exists():
        (project_root / 'README.md').write_text(README_MD)
        print(f"📄 创建说明文件: README.md")
    
    # setup.py
    if not (project_root / 'setup.py').exists():
        (project_root / 'setup.py').write_text(SETUP_PY)
        print(f"📄 创建安装文件: setup.py")
    
    # .gitignore
    if not (project_root / '.gitignore').exists():
        (project_root / '.gitignore').write_text(GITIGNORE)
        print(f"📄 创建Git忽略文件: .gitignore")


def print_next_steps():
    \"\"\"打印后续步骤\"\"\"
    
    print(\"\\n\" + \"=\"*60)
    print(\"🎉 Colony Detection SAM 2.0 项目结构创建完成!\")
    print(\"=\"*60)
    
    print(\"\\n📋 下一步操作:\")
    print(\"1. 安装依赖包:\")
    print(\"   pip install -r requirements.txt\")
    
    print(\"\\n2. 下载SAM模型权重:\")
    print(\"   - 下载 vit_b 模型到 models/ 目录\")
    print(\"   - 下载地址: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\")
    
    print(\"\\n3. 将重构后的代码文件复制到对应目录:\")
    print(\"   - SAM模型代码 → colony_analysis/core/sam_model.py\")
    print(\"   - 检测器代码 → colony_analysis/core/detection.py\")
    print(\"   - 分析器代码 → colony_analysis/analysis/colony.py\")
    print(\"   - 特征提取代码 → colony_analysis/analysis/features.py\")
    print(\"   - 评分系统代码 → colony_analysis/analysis/scoring.py\")
    print(\"   - 配置管理代码 → colony_analysis/config/settings.py\")
    print(\"   - 工具类代码 → colony_analysis/utils/ 对应文件\")
    print(\"   - 分析管道代码 → colony_analysis/pipeline.py\")
    
    print(\"\\n4. 测试运行:\")
    print(\"   python main.py --help\")
    
    print(\"\\n5. 基本使用:\")
    print(\"   python main.py --image test_image.jpg --output results/\")
    
    print(\"\\n📁 项目结构:\")
    print(\"Colony_detection_SAM_2.0/\")
    print(\"├── main.py                    # 主入口文件\")
    print(\"├── config.yaml               # 配置文件\")
    print(\"├── requirements.txt          # 依赖列表\")
    print(\"├── colony_analysis/          # 主包目录\")
    print(\"│   ├── pipeline.py           # 分析管道\")
    print(\"│   ├── config/               # 配置管理\")
    print(\"│   ├── core/                 # 核心功能\")
    print(\"│   ├── analysis/             # 分析模块\")
    print(\"│   └── utils/                # 工具模块\")
    print(\"├── tests/                    # 测试目录\")
    print(\"├── models/                   # 模型存放目录\")
    print(\"└── examples/                 # 示例目录\")


def main():
    \"\"\"主函数\"\"\"
    
    print(\"🚀 开始创建 Colony Detection SAM 2.0 项目结构...\")
    
    try:
        # 创建目录结构
        create_project_structure()
        
        # 创建配置文件
        create_config_files()
        
        # 打印后续步骤
        print_next_steps()
        
    except Exception as e:
        print(f\"❌ 创建项目结构时发生错误: {e}\")
        sys.exit(1)


if __name__ == \"__main__\":
    main()
"""
