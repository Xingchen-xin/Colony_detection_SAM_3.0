# ============================================================================
# colony_analysis/core/mmm_front.py - MMM Front analysis
# ============================================================================

from argparse import Namespace
from pathlib import Path


def mmm_front_analysis(image_path: str, save_folder: str):
    """MMM Front 图像分析逻辑

    调用 ``AnalysisPipeline`` 执行完整流程。
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
        orientation="front",
        replicate=None,
    )

    pipeline = AnalysisPipeline(args)
    return pipeline.run()
