# Colony Detection SAM 2.0

基于 Segment Anything Model (SAM) 的链霉菌菌落检测和分析工具。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```
依赖列表现在包含 `openpyxl`，用于将配对结果导出为 Excel 文件。

### 运行批量分析
```bash
python main.py --input-dir Image_input --output results [--device cpu|cuda]
```
默认在检测到 GPU 时使用 `cuda`，可通过 `--device cpu` 强制在 CPU 上运行。
或直接调用管道：
```bash
cd colony_analysis
python pipeline.py -i ../Image_input -o ../results
```
- 批处理过程会显示实时进度条和每一步耗时，便于跟踪整体进度。
- 日志输出与进度条兼容，调试信息不会打断动态进度显示。

### 文件名解析规则
图像文件名需包含样本名称、培养基、拍摄角度以及可选的重复编号：
```
<sample_name>_<medium>_<orientation>[_<replicate>]
```
- `<medium>` 为 `R5` 或 `MMM`
- `<orientation>` 为 `Back` 或 `Front`
- `<replicate>` 可选，两位数字，缺省为 `01`

配置文件 `config.yaml` 允许按培养基和拍摄角度覆盖参数，例如：

```yaml
medium_params:
  mm:
    back:
      detection:
        background_filter: false
```

### 结果目录结构
```
results/
  <sample_name>/
    <MEDIUM>/
      replicate_01/
        Front/
          annotated_Front.png
          stats_Front.txt
        Back/
          annotated_Back.png
          stats_Back.txt
        combined/
          combined_stats.txt
      replicate_02/
        ...
      summary/
        all_replicates.csv
        summary_stats.txt
```

### 开发指南
- 新的分析函数位于 `colony_analysis/core/`。
- 通用工具如 `collect_all_images` 和 `parse_filename` 位于 `colony_analysis/utils/file_utils.py`。
- 批量调度逻辑在 `colony_analysis/pipeline.py` 的 `batch_medium_pipeline` 函数中。

### 更多文档
- 轻量级分割与评估管线请参见 [docs/segmentation_pipeline.md](docs/segmentation_pipeline.md)

欢迎提交 Issue 和 Pull Request!
