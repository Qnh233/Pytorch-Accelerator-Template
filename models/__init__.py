# 模型模块
from .resnet import ResNet
from .transformer import Transformer
# from .vit import VisionTransformer


def get_model(name: str):
    """根据名称获取模型类"""
    model_registry = {
        "ResNet": ResNet,
        "Transformer": Transformer,
        # "VisionTransformer": VisionTransformer,
    }

    if name not in model_registry:
        raise ValueError(f"Unknown model: {name}. Available: {list(model_registry.keys())}")

    return model_registry[name]


def create_model(model_config: dict):
    """根据配置创建模型实例"""
    model_class = get_model(model_config["name"])
    params = model_config.get("params", {}) or {}  # 确保params不会是None
    return model_class(**params)
