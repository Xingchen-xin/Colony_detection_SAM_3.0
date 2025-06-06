# ============================================================================
# colony_analysis/core/r5_front.py - R5 Front analysis
# ============================================================================

from argparse import Namespace
from pathlib import Path


def r5_front_analysis(image_path: str, save_folder: str):
    """R5 Front 图像分析逻辑

    该函数构造 ``AnalysisPipeline`` 所需的参数并直接调用 ``run``。
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
        orientation="front",
        replicate=None,
    )

    pipeline = AnalysisPipeline(args)
    return pipeline.run()
