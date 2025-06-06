# ============================================================================
# colony_analysis/core/mmm_back.py - MMM Back analysis
# ============================================================================

from argparse import Namespace
from pathlib import Path


def mmm_back_analysis(image_path: str, save_folder: str):
    """MMM Back 图像分析逻辑

    调用 ``AnalysisPipeline`` 完成菌落检测与分析。
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
        medium="mmm",
        orientation="back",
        replicate=None,
    )

    pipeline = AnalysisPipeline(args)
    return pipeline.run()
