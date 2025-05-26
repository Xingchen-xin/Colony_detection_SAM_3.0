# ============================================================================
# 15. README.md
# ============================================================================

README_MD = """# Colony Detection SAM 2.0

基于Segment Anything Model (SAM)的链霉菌菌落检测和分析工具

## 功能特点

- 🔬 高精度菌落分割和检测
- 📊 全面的形态学特征分析  
- 🎨 代谢产物识别和定量
- 📈 智能评分和表型分类
- 🔧 支持96孔板自动识别
- 📋 丰富的输出格式和可视化

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载SAM模型

下载相应的SAM模型权重文件到 `models/` 目录：

- [vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)  
- [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

### 3. 基本使用

```bash
# 基本分析
python main.py --image your_image.jpg --output results/

# 高级分析模式
python main.py --image your_image.jpg --output results/ --advanced --debug

# 96孔板模式
python main.py --image plate.jpg --well-plate --mode grid
```

## 项目结构

```
Colony_detection_SAM_2.0/
├── main.py                    # 主入口文件
├── config.yaml               # 配置文件
├── colony_analysis/          # 主包
│   ├── config/              # 配置管理
│   ├── core/                # 核心功能
│   ├── analysis/            # 分析模块
│   ├── utils/               # 工具模块
│   └── pipeline.py          # 分析管道
├── tests/                   # 测试文件
├── models/                  # 模型权重存放目录
└── examples/               # 示例和文档
```

## 配置说明

主要配置参数在 `config.yaml` 中：

- `detection`: 检测相关参数
- `sam`: SAM模型参数
- `analysis`: 分析功能参数
- `output`: 输出格式参数

## 输出说明

程序会在输出目录生成：

- `results/analysis_results.csv`: 分析结果表格
- `colonies/`: 单个菌落图像
- `visualizations/`: 检测和分析可视化
- `reports/`: 分析报告

## 开发指南

### 环境设置

```bash
# 安装开发依赖
pip install -r requirements.txt

# 代码格式化
black .
isort .

# 运行测试
pytest tests/
```

## 版本历史

- **v2.0**: 架构重构，模块化设计
- **v1.0**: 初始版本

## 许可证

Apache 2.0 License

## 贡献指南

欢迎提交Issue和Pull Request！
"""