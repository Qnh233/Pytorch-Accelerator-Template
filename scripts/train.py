#!/usr/bin/env python3
import argparse
import logging
import pdb
import random
import traceback
from datetime import datetime, timedelta
import sys
import os
import torch
import numpy as np
from accelerate.logging import get_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加这行

# 添加项目根目录到Python路径
from utils.config import load_config
from models import create_model
from datasets import create_dataset
from training.trainer import AccelerateTrainer
# def exception_hook(exc_type, exc_value, exc_traceback):
#     # 获取当前时间并加上8小时 (东八区)
#     timestamp = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}],异常发生")
#     traceback.print_exception(exc_type, exc_value, exc_traceback)
#     sys.stdout.flush()
#     pdb.post_mortem(exc_traceback)
#
# sys.excepthook = exception_hook
# 设置accelerate logger的级别和处理器
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)  # 设置日志级别
handler = logging.StreamHandler()  # 添加控制台处理器
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.propagate = False  # 防止重复日志
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--overrides", nargs="*", help="配置覆盖参数，格式: key=value")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config, args.overrides)

    set_seed(config["experiment"]["seed"])

    # 创建输出目录
    os.makedirs(config["experiment"]["output_dir"], exist_ok=True)

    # 创建模型
    model = create_model(config["model"])

    # 创建数据集
    train_dataset = create_dataset(config["data"], split="train")
    val_dataset = create_dataset(config["data"], split="val")

    # 创建训练器并开始训练
    trainer = AccelerateTrainer(config)
    trainer.setup_training(model, train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    main()