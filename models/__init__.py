# 模型模块
import os
from utils.registry import MODELS, import_modules_from_folder

# Automatically scan and import models
# This assumes this __init__.py is in .../models/
# We want to scan the current directory.
import_modules_from_folder(os.path.dirname(__file__), __package__)


def create_model(model_config: dict):
    """根据配置创建模型实例"""
    model_class = MODELS.get(model_config["name"])
    params = model_config.get("params", {}) or {}  # 确保params不会是None
    return model_class(**params)
