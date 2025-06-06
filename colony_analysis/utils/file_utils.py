# ============================================================================
# colony_analysis/utils/file_utils.py - 通用文件工具
# ============================================================================

from pathlib import Path
from typing import List, Tuple


def collect_all_images(input_folder: str) -> List[Path]:
    """递归收集指定目录下所有支持的图片文件"""
    img_dir = Path(input_folder)
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    images: List[Path] = []
    for ext in exts:
        images.extend(img_dir.rglob(f"*{ext}"))
    return images


def parse_filename(stem: str) -> Tuple[str, str, str, str]:
    """解析文件名, 返回 (sample_name, medium, orientation, replicate).

    此函数兼容類似 ``Lib96_Cumate100_@MMM_Front20250401_09234224_1`` 的文件名,
    即培養基與拍攝角度可能與其他字符串連在一起。解析規則如下:

    - 若最後一段為純數字, 視為生物學重複編號, 否則默認 ``01``。
    - ``medium`` 通過在文件名中搜索 ``r5`` 或 ``mmm`` 決定(不分大小寫)。
    - ``orientation`` 通过搜索 ``back`` 或 ``front`` 決定(不分大小寫)。
    - ``sample_name`` 為去掉擴展名及重複編號後的剩餘部分, 不移除培養基或角度
      關鍵字, 以保留原始名稱。
    """

    parts = stem.split("_")
    replicate = "01"
    if parts and parts[-1].isdigit():
        replicate = parts[-1].zfill(2)
        stem_no_rep = "_".join(parts[:-1])
    else:
        stem_no_rep = stem

    lower = stem_no_rep.lower()
    medium = ""
    if "r5" in lower:
        medium = "r5"
    elif "mmm" in lower:
        medium = "mmm"

    orientation = ""
    if "back" in lower:
        orientation = "back"
    elif "front" in lower:
        orientation = "front"

    sample_name = stem_no_rep
    return sample_name, medium, orientation, replicate

