# 模型模块
import os
from utils.registry import MODELS, import_modules_from_folder

# Automatically scan and import models, backbones, heads, architectures
import_modules_from_folder(os.path.dirname(__file__), __package__)


def create_model(model_config: dict):
    """根据配置创建模型实例"""
    # model_config now contains "name" which refers to the Architecture (e.g., ImageClassifier)
    # and other keys like "backbone", "head" which are passed as kwargs.

    name = model_config.get("name")
    if not name:
        raise ValueError("Model config must contain 'name' field.")

    model_class = MODELS.get(name)

    # We copy config to avoid modifying it
    params = model_config.copy()
    params.pop("name")

    # Some older configs might still have "params" nested.
    # If "params" exists and no other keys (like backbone), assume old style?
    # Or strict new style?
    # Let's support both: if "params" is present, use it as kwargs.
    # If not, use the rest of config as kwargs.

    if "params" in params and len(params) == 1:
        kwargs = params["params"]
    else:
        kwargs = params

    return model_class(**kwargs)
