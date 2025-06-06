# ============================================================================
# colony_analysis/core/mmm_back.py - MMM Back analysis
# ============================================================================

import cv2
import numpy as np
from pathlib import Path


def mmm_back_analysis(image_path: str, save_folder: str):
    """MMM_Back 图像分析逻辑"""
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    # TODO: 对 l_channel 应用 CLAHE，再分割 b_channel/r_channel
    annotated = img.copy()
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(save_folder) / "annotated_Back.png"), annotated)
    # with open(Path(save_folder) / "stats_Back.txt", "w") as f:
    #     f.write("...统计数据...")
