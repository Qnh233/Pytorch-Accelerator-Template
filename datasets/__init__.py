# 数据集模块
import os
from utils.registry import DATASETS, import_modules_from_folder

# Automatically scan and import datasets
import_modules_from_folder(os.path.dirname(__file__), __package__)


def create_dataset(dataset_config: dict, split: str = "train"):
    """根据配置创建数据集实例"""
    dataset_class = DATASETS.get(dataset_config["name"])
    params = dataset_config.get("data_params", {}) or {}  # 确保params不会是None
    return dataset_class(split=split, **params)
