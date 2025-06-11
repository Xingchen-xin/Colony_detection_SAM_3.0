import os
import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config(medium: str, side: str):
    base_dir = os.path.join(os.path.dirname(__file__), 'config')
    config = {}
    common_path = os.path.join(base_dir, 'common.yaml')
    if os.path.exists(common_path):
        config.update(load_yaml(common_path) or {})

    medium_file = os.path.join(base_dir, f'{medium}.yaml')
    if os.path.exists(medium_file):
        medium_cfg = load_yaml(medium_file) or {}
        specific_cfg = medium_cfg.get(side, {})
        config.update(specific_cfg)
    else:
        raise FileNotFoundError(f'配置文件不存在: {medium_file}')
    return config
