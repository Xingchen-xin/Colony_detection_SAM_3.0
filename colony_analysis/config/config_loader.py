import os
import yaml
from typing import Dict, Any

def parse_condition_from_path(path: str) -> (str, str):
    """
    从文件路径或名称推断实验条件。
    返回 (medium, orientation)，例如 ("r5","back")。
    """
    fname = os.path.basename(path).lower()
    if "r5" in fname:
        medium = "r5"
    elif "mm" in fname:
        medium = "mm"
    else:
        medium = "default"
    if "back" in fname:
        orientation = "back"
    elif "front" in fname:
        orientation = "front"
    else:
        orientation = "default"
    return medium, orientation

class ConfigLoader:
    """
    多文件条件化配置加载器。
    目录结构示例:
      configs/
        default/
          default.yaml
        r5/
          front.yaml
          back.yaml
        mm/
          front.yaml
          back.yaml
    """

    def __init__(self, config_base_dir: str):
        self.base_dir = config_base_dir
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self):
        """将所有yaml配置预读到内存缓存中"""
        for medium in os.listdir(self.base_dir):
            medium_dir = os.path.join(self.base_dir, medium)
            if not os.path.isdir(medium_dir): continue
            for fname in os.listdir(medium_dir):
                if fname.endswith((".yaml", ".yml")):
                    key = f"{medium}/{fname[:-5]}"
                    try:
                        with open(os.path.join(medium_dir, fname), "r", encoding="utf-8") as f:
                            self._cache[key] = yaml.safe_load(f) or {}
                    except Exception:
                        continue

    def load_for_image(self, image_path: str) -> Dict[str, Any]:
        """根据图像路径、名称自选配置"""
        medium, orientation = parse_condition_from_path(image_path)
        cfg: Dict[str, Any] = {}
        # 合并 default/default
        cfg.update(self._cache.get("default/default", {}))
        # 合并 medium/default
        cfg.update(self._cache.get(f"{medium}/default", {}))
        # 合并 medium/orientation
        cfg.update(self._cache.get(f"{medium}/{orientation}", {}))
        return cfg