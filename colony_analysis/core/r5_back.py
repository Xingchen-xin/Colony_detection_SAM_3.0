# ============================================================================
# colony_analysis/core/r5_back.py - R5 Back analysis
# ============================================================================

from argparse import Namespace
from pathlib import Path


def r5_back_analysis(image_path: str, save_folder: str):
    """R5 Back 图像分析逻辑

    与 ``r5_front_analysis`` 类似，调用 ``AnalysisPipeline`` 处理图像。
    """

    from ..pipeline import AnalysisPipeline

    args = Namespace(
        image=image_path,
        output=str(Path(save_folder)),
        mode="auto",
        model="vit_b",
        advanced=False,
        debug=False,
        config=None,
        min_area=2000,
        well_plate=False,
        rows=8,
        cols=12,
        verbose=False,
        medium="r5",
        orientation="back",
        replicate=None,
    )

    pipeline = AnalysisPipeline(args)
    return pipeline.run()
