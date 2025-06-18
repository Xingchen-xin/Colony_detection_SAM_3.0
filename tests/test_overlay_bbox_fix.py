import numpy as np
from pathlib import Path
from colony_analysis.utils.visualization import ImprovedVisualizer


def test_overlay_bbox_autofix(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.ones((20, 20), dtype=bool)
    colonies = [{"bbox": (10, 30, 30, 50), "mask": mask}]
    viz = ImprovedVisualizer(str(tmp_path))
    # Provide incorrectly ordered bbox (x1,y1,x2,y2)
    colonies_bad = [{"bbox": (30, 10, 50, 30), "mask": mask}]
    try:
        viz.overlay_masks(img, [mask], Path(tmp_path), colonies_bad)
    except Exception as e:
        raise AssertionError(f"overlay_masks raised {e}")
