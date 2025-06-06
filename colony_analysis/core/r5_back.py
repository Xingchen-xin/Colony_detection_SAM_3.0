# ============================================================================
# colony_analysis/core/r5_back.py - R5 Back analysis
# ============================================================================

import cv2
import numpy as np
from pathlib import Path


def r5_back_analysis(image_path: str, save_folder: str):
    """R5_Back 图像分析逻辑"""
    img = cv2.imread(image_path)
    b_channel, g_channel, r_channel = cv2.split(img)
    # TODO: 在这里加入 R5_Back 色素分割与统计逻辑
    annotated = img.copy()
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(save_folder) / "annotated_Back.png"), annotated)
    # with open(Path(save_folder) / "stats_Back.txt", "w") as f:
    #     f.write("...统计数据...")
