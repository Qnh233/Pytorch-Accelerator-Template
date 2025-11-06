# 数据集模块
from .cifar10 import CIFAR10Dataset
from .mnist import MNISTDataset

# 数据集注册
def get_dataset(name: str):
    """根据名称获取数据集类"""
    dataset_registry = {
        "CIFAR10": CIFAR10Dataset,
        # "ImageNet": ImageNetDataset,
        # "Custom": CustomDataset,
        "MNIST": MNISTDataset,
    }

    if name not in dataset_registry:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(dataset_registry.keys())}")

    return dataset_registry[name]


def create_dataset(dataset_config: dict, split: str = "train"):
    """根据配置创建数据集实例"""
    dataset_class = get_dataset(dataset_config["name"])
    params = dataset_config.get("data_params", {}) or {}  # 确保params不会是None
    return dataset_class(split=split, **params)
