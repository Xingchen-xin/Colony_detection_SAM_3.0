# Colony Segmentation and Evaluation Pipeline

此文档概述了在项目中引入轻量级菌落分割模型以及评估流程的设计思路。内容包括模型选择、与现有系統的集成方式、自动化评估、配置化管理以及核心函数接口示例。

## 1. 模型选择

在 GPU 资源有限的情况下，可选的轻量模型包括：

- **U-Net**：经典卷积下采样/上采样结构，可在 Segmentation Models PyTorch(SMP) 中加载预训练编码器并快速微调。
- **FastSAM**：基于 YOLOv8 分割头的 Segment Anything 轻量实现，无需提示即可分割图像中所有对象，可作为兜底方案。
- **SegFormer**：NVIDIA 提出的高效 Transformer 架构，B0/B1 等小型号仅数百万参数，适合在不同培养基和成像条件下保持鲁棒。

## 2. 与现有系统集成

将不同分割模型封装为统一接口 `segment(image) -> masks`，然后在 SAM3.0 的 `fallback` 或 `ensemble` 流程中按需调用。可先运行主模型，若判断为漏检则调用兜底模型补充；或多模型并行推理后合并结果。

## 3. 评估流程

评估管线读取图像路径以及培养基和拍摄方向，根据配置加载对应模型运行分割，并与弱标注真值比对。指标包括 Precision、Recall、F1、IoU 等。每次评估都会输出日志（JSON）并在 CSV 表中汇总，便于比较不同模型和参数组合的效果。

```python
# 示例入口接口
from pipeline import evaluate_colonies
summary = evaluate_colonies("images/sample_MMM_front.jpg", "config.yaml")
```

## 4. 配置化管理

配置文件采用 YAML 格式，为不同培养基与拍摄方向指定默认模型及参数，同时允许在 `experiments` 中列出额外的模型组合进行对比。更改配置即可添加新模型或调整阈值，无需修改代码。

```yaml
# config.yaml 中的示例片段
defaults:
  "R5_front":
    model: "FastSAM"
    weights: "./weights/FastSAM.pt"
    threshold: 0.5
```

## 5. 模块与接口

- `config.py`：加载与解析 YAML 配置。
- `models/`：实现 `BaseModel` 及 `FastSAMModel`、`UNetModel`、`SegFormerModel` 等子类。
- `metrics.py`：提供掩膜匹配与指标计算函数。
- `pipeline.py`：实现 `evaluate_colonies`，串联读取配置、模型推理、评估和日志输出。

通过模块化设计，后续可轻松扩展新的分割模型或指标，并利用 Codex 自动生成剩余实现。
