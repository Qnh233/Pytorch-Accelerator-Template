from accelerate import Accelerator
import torch
import torch.nn as nn
from typing import Dict, Any, List

from accelerator import create_tracker
from models.losses import create_loss
from .callbacks import get_callback

class AccelerateTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
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

        # 初始化状态
        self.epoch = 0
        self.global_step = 0
        self.best_metric = None
        # 设置损失函数
        self.criterion = self._setup_loss()
        # 设置回调函数
        self.callbacks = self._setup_callbacks()

    def _setup_loss(self):
        """设置损失函数"""
        if "loss" in self.config:
            loss_config = self.config["loss"]
            return create_loss(loss_config)
        else:
            # 默认使用交叉熵损失
            return torch.nn.CrossEntropyLoss()
    def _setup_callbacks(self) -> List:
        """根据配置设置回调函数"""
        callbacks = []
        for callback_config in self.config.get("callbacks", []):
            if isinstance(callback_config, str):
                # 简单字符串配置
                callback_class = get_callback(callback_config)
                callbacks.append(callback_class(self.config))
            elif isinstance(callback_config, dict):
                # 带参数的配置
                callback_name = callback_config["name"]
                callback_class = get_callback(callback_name)
                # 合并全局配置和回调特定配置
                callback_params = {**self.config, **callback_config.get("params", {})}
                callbacks.append(callback_class(callback_params))
        return callbacks

    def setup_training(self, model, train_dataset, val_dataset):
        """设置训练组件"""
        # 数据加载器
        data_config = self.config["data"]["data_loader_params"]
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=data_config["batch_size"],
            shuffle=True,
            num_workers=data_config.get("num_workers", 0)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=data_config["batch_size"],
            shuffle=False,
            num_workers=data_config.get("num_workers", 0)
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
            images, labels = batch  # 假设batch是(image, label)元组
            self.optimizer.zero_grad()

            # 前向传播
            with self.accelerator.autocast():
                outputs = self.model(images)
                # 计算损失
                if hasattr(outputs, 'loss'):
                    # 如果模型返回损失（如 transformers）
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    # 如果模型返回包含损失的字典
                    loss = outputs['loss']
                else:
                    # 使用配置的损失函数
                    loss = self.criterion(outputs, labels)

            # 反向传播
            self.accelerator.backward(loss)
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
                images, labels = batch  # batch是(image, label)元组
                outputs = self.model(images)
                # 计算损失
                if hasattr(outputs, 'loss'):
                    # 如果模型返回损失（如 transformers）
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    # 如果模型返回包含损失的字典
                    loss = outputs['loss']
                else:
                    # 使用配置的损失函数
                    loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                # 收集预测和标签供回调使用
                all_outputs.append(outputs)
                all_labels.append(labels)

        avg_loss = total_loss / len(self.val_loader)

        # 记录指标
        if self.accelerator.is_main_process:
            self.accelerator.log({"val_loss": avg_loss})

        return {
                "val_loss": avg_loss,
                "outputs": torch.cat(all_outputs),
                "labels": torch.cat(all_labels)
                }