from accelerate import Accelerator
import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Union
import os
import json
import subprocess
import random
import numpy as np
import logging

from accelerator import create_tracker
from utils.registry import LOSSES, CALLBACKS
import models.losses # Ensure standard losses are registered
import training.callbacks # Ensure callbacks are registered
from training.sanity_checks import check_data_leakage, visual_sanity_check, check_system_config

class AccelerateTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Get Git Commit Hash
        try:
            self.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception:
            self.git_hash = "unknown"

        self.config["experiment"]["git_hash"] = self.git_hash

        # Create output directory explicitly if not exists (accelerator might do it but we need to save config)
        os.makedirs(config["experiment"]["output_dir"], exist_ok=True)

        # Save configuration
        self._save_config()

        self.tracker = create_tracker(config["training"].get("log_custom_with", None))
        if self.tracker is not None:
            self.tracker.store_init_configuration(config)
        self.accelerator = Accelerator(
            mixed_precision=config["training"].get("mixed_precision", "no"),
            log_with=self.tracker if self.tracker is not None else config["training"].get("log_with", None),
            project_dir=config["experiment"]["output_dir"]
        )
        if self.tracker is None:
            # 展平配置字典
            def flatten_dict(d, parent_key='', sep='.'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    elif isinstance(v, list):
                        items.extend(flatten_dict({f"{new_key}{sep}{i}": item for i, item in enumerate(v)}).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
            flattened_config = flatten_dict(config)
            self.accelerator.init_trackers(config["experiment"]["name"], config=flattened_config)

        # Log Git Hash
        if self.accelerator.is_main_process:
            print(f"🔗 Git Commit Hash: {self.git_hash}")

        # 初始化状态
        self.epoch = 0
        self.global_step = 0
        self.best_metric = None
        # 设置损失函数
        self.criterion = self._setup_loss()
        # 设置回调函数
        self.callbacks = self._setup_callbacks()

    def _save_config(self):
        """Save the full configuration to the output directory."""
        config_path = os.path.join(self.config["experiment"]["output_dir"], "config_dump.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save config dump: {e}")

    def _setup_loss(self):
        """设置损失函数"""
        if "loss" in self.config:
            loss_config = self.config["loss"]
            # return create_loss(loss_config) # Deprecated
            loss_class = LOSSES.get(loss_config["name"])
            params = loss_config.get("params", {}) or {}
            return loss_class(**params)
        else:
            # 默认使用交叉熵损失
            return torch.nn.CrossEntropyLoss()

    def _setup_callbacks(self) -> List:
        """根据配置设置回调函数"""
        callbacks = []
        for callback_config in self.config.get("callbacks", []):
            if isinstance(callback_config, str):
                # 简单字符串配置
                callback_class = CALLBACKS.get(callback_config)
                callbacks.append(callback_class(self.config))
            elif isinstance(callback_config, dict):
                # 带参数的配置
                callback_name = callback_config["name"]
                callback_class = CALLBACKS.get(callback_name)
                # 合并全局配置和回调特定配置
                callback_params = {**self.config, **callback_config.get("params", {})}
                callbacks.append(callback_class(callback_params))
        return callbacks

    @staticmethod
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def setup_training(self, model, train_dataset, val_dataset):
        """设置训练组件"""
        # Generator for DataLoader reproducibility
        g = torch.Generator()
        g.manual_seed(self.config["experiment"]["seed"])

        # 数据加载器
        data_config = self.config["data"]["data_loader_params"]
        # Default num_workers to CPU count if not specified
        num_workers = data_config.get("num_workers")
        if num_workers is None:
            num_workers = os.cpu_count() or 1

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=data_config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=data_config.get("pin_memory", True),
            worker_init_fn=self._seed_worker,
            generator=g
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=data_config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=data_config.get("pin_memory", True),
            worker_init_fn=self._seed_worker,
            generator=g
        )

        # 优化器
        optimizer_config = self.config["optimizer"]
        optimizer_class = getattr(torch.optim, optimizer_config["name"])
        optimizer = optimizer_class(model.parameters(), **optimizer_config["params"])

        # 学习率调度器
        scheduler = None
        if "scheduler" in self.config:
            scheduler_config = self.config["scheduler"]
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_config["name"])
            scheduler = scheduler_class(optimizer, **scheduler_config["params"])

        # 准备训练组件
        self.model, self.optimizer, self.train_loader, self.val_loader = \
            self.accelerator.prepare(model, optimizer, train_loader, val_loader)

        if scheduler:
            self.scheduler = self.accelerator.prepare(scheduler)
        else:
            self.scheduler = None

        return self.model, self.optimizer, self.train_loader, self.val_loader

    def train(self):
        """执行训练"""
        # Run Sanity Checks
        if self.accelerator.is_main_process:
             print("🛡️ Running Sanity Checks...")
             check_system_config(self.config)
             # Note: accelerator.prepare might wrap loader, but usually preserves dataset attribute
             check_data_leakage(self.train_loader.dataset, self.val_loader.dataset)
             visual_sanity_check(self.train_loader, os.path.join(self.config["experiment"]["output_dir"], "sanity_check"))
             print("✅ Sanity Checks Completed.")

        training_config = self.config["training"]

        # 调用训练开始回调
        for callback in self.callbacks:
            callback.on_train_start(self)

        for epoch in range(training_config["epochs"]):
            self.epoch = epoch

            # 调用epoch开始回调
            for callback in self.callbacks:
                callback.on_epoch_start(self)

            # 训练一个epoch
            train_metrics = self._train_epoch()

            # 验证
            if epoch % training_config["eval_every"] == 0:
                val_metrics = self._validate()
            else:
                val_metrics = {}
            train_loss = train_metrics.get("train_loss", 0)
            val_loss = val_metrics.get("val_loss", 0)
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            # 调用epoch结束回调
            for callback in self.callbacks:
                should_stop = callback.on_epoch_end(self, train_metrics, val_metrics)
                if should_stop:
                    break

        # 调用训练结束回调
        for callback in self.callbacks:
            callback.on_train_end(self)
        self.accelerator.end_training()

    def _train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # 调用batch开始回调
            for callback in self.callbacks:
                callback.on_batch_start(self, batch)

            self.optimizer.zero_grad()

            # 前向传播
            with self.accelerator.autocast():
                # Generic batch handling
                if isinstance(batch, dict):
                    # Unpack dict to model arguments if possible, or pass as kwargs
                    # NOTE: Some models might expect specific arg names.
                    # Assuming here that if it's a dict, we pass kwargs.
                    # BUT `accelerator` might wrap the model, so we need to be careful.
                    # Usually `model(**batch)` works for transformers.

                    # However, batch might contain 'labels' which some models take, others don't.
                    # Let's try passing generic `**batch` if it's a dict.

                    # To be safe for custom models that take (x), we might need config.
                    # For now, let's assume if it is a dict, model knows how to handle it.
                    try:
                         outputs = self.model(**batch)
                    except TypeError:
                         # Fallback: maybe model expects 'input_ids' etc but batch has other things
                         # Or model expects positional args.
                         # This is risky. Let's try to extract 'pixel_values' or 'input_ids' or just 'x'?
                         # For this codebase (CV focus), it might be (images, labels).
                         # If batch is dict, maybe it has 'image' and 'label'?
                         if "image" in batch:
                             outputs = self.model(batch["image"])
                         else:
                             raise ValueError("Batch is a dict but no 'image' key found and model(**batch) failed.")
                elif isinstance(batch, (list, tuple)):
                    images = batch[0]
                    outputs = self.model(images)
                else:
                    outputs = self.model(batch)

                # 计算损失
                if hasattr(outputs, 'loss'):
                    # 如果模型返回损失（如 transformers）
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    # 如果模型返回包含损失的字典
                    loss = outputs['loss']
                else:
                    # 使用配置的损失函数
                    # Need labels
                    labels = None
                    if isinstance(batch, (list, tuple)) and len(batch) > 1:
                        labels = batch[1]
                    elif isinstance(batch, dict):
                        labels = batch.get("label", batch.get("labels"))

                    if labels is None:
                         raise ValueError("Could not find labels for loss calculation.")

                    loss = self.criterion(outputs, labels)

            # 反向传播
            self.accelerator.backward(loss)

            # Check Loss NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logging.error(f"Loss is {loss.item()} at step {self.global_step}")
                raise ValueError("Loss is NaN or Inf. Stopping training.")

            # Monitor Gradient Norm
            if self.config["training"].get("log_grad_norm", True):
                 # Compute norm
                 total_norm = 0.0
                 for p in self.model.parameters():
                     if p.grad is not None:
                         total_norm += p.grad.detach().data.norm(2).item() ** 2
                 total_norm = total_norm ** 0.5
                 if self.accelerator.is_main_process:
                     self.accelerator.log({"grad_norm": total_norm}, step=self.global_step)

            # Gradient Clipping
            if "max_grad_norm" in self.config["training"]:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config["training"]["max_grad_norm"])

            self.optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item
            self.global_step += 1
            # 记录日志
            if self.accelerator.is_main_process:
                self.accelerator.log({"train_loss": loss_item})

            # 调用batch结束回调
            for callback in self.callbacks:
                callback.on_batch_end(self, batch, outputs, loss)

        return {"train_loss": total_loss / len(self.train_loader)}

    def _validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Generic batch handling
                if isinstance(batch, dict):
                    try:
                         outputs = self.model(**batch)
                    except TypeError:
                         if "image" in batch:
                             outputs = self.model(batch["image"])
                         else:
                             raise ValueError("Batch is a dict but no 'image' key found and model(**batch) failed.")
                elif isinstance(batch, (list, tuple)):
                    images = batch[0]
                    outputs = self.model(images)
                else:
                    outputs = self.model(batch)

                # 计算损失
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    labels = None
                    if isinstance(batch, (list, tuple)) and len(batch) > 1:
                        labels = batch[1]
                    elif isinstance(batch, dict):
                        labels = batch.get("label", batch.get("labels"))

                    if labels is None:
                         raise ValueError("Could not find labels for loss calculation.")

                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                # 收集预测和标签供回调使用
                if not hasattr(outputs, 'loss') and not (isinstance(outputs, dict) and 'loss' in outputs):
                     # If model returned loss, outputs might be the loss itself or Object.
                     # If we computed loss manually using criterion, outputs is predictions.
                     all_outputs.append(outputs)

                # If model returned loss, we might not have predictions easy to concat?
                # For now assume if we use criterion, we have outputs.

                if isinstance(batch, (list, tuple)) and len(batch) > 1:
                    all_labels.append(batch[1])
                elif isinstance(batch, dict):
                    l = batch.get("label", batch.get("labels"))
                    if l is not None:
                        all_labels.append(l)

        avg_loss = total_loss / len(self.val_loader)

        # 记录指标
        if self.accelerator.is_main_process:
            self.accelerator.log({"val_loss": avg_loss})

        result = {"val_loss": avg_loss}
        if all_outputs:
            result["outputs"] = torch.cat(all_outputs)
        if all_labels:
            result["labels"] = torch.cat(all_labels)

        return result
