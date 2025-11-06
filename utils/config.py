# 配置加载文件
# 从配置文件中加载配置
import yaml
import os
from typing import Dict, Any


def load_config(config_path: str, overrides: list = None) -> Dict[str, Any]:
    """加载配置文件，支持嵌套引用和命令行覆盖"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 处理文件引用
    config = _resolve_file_references(config, os.path.dirname(config_path))

    # 应用命令行覆盖
    if overrides:
        config = _apply_overrides(config, overrides)

    return config


def _resolve_file_references(config: Dict, base_dir: str) -> Dict:
    """解析配置文件中的_file_引用"""
    for key, value in config.items():
        if isinstance(value, dict) and '_file_' in value:
            file_path = os.path.join(base_dir, value['_file_'])
            with open(file_path, 'r') as f:
                file_config = yaml.safe_load(f)
            # 合并配置
            config[key] = {**file_config, **{k: v for k, v in value.items() if k != '_file_'}}
        elif isinstance(value, dict):
            config[key] = _resolve_file_references(value, base_dir)

    return config


def _apply_overrides(config: Dict, overrides: list) -> Dict:
    """应用命令行参数覆盖"""
    for override in overrides:
        key_path, value = override.split('=')
        keys = key_path.split('.')

        # 导航到目标位置
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # 设置值（尝试类型转换）
        last_key = keys[-1]
        if last_key in current:
            original_type = type(current[last_key])
            try:
                current[last_key] = original_type(value)
            except ValueError:
                current[last_key] = value
        else:
            current[last_key] = value

    return config