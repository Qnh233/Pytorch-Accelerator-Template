import os
import torch
import numpy as np
from typing import Dict, Any, Optional
import json


class Callback:
    """回调基类"""

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        pass

    def on_batch_start(self, trainer, batch):
        pass

    def on_batch_end(self, trainer, batch, outputs, loss):
        pass

    def get_logged_metrics(self, trainer):
        """获取当前记录的指标"""
        if hasattr(trainer.accelerator, 'log_records'):
            return trainer.accelerator.log_records
        return {}
class ModelCheckpoint(Callback):
    """模型检查点回调"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.save_dir = config["experiment"]["output_dir"]
        self.save_every = config["training"].get("save_every", 10)
        self.save_best = config["training"].get("save_best", True)
        self.monitor = config["training"].get("monitor_metric", "val_loss")
        self.mode = config["training"].get("monitor_mode", "min")  # "min" or "max"

        self.best_metric = None
        self.best_epoch = None

        # 创建保存目录
        os.makedirs(os.path.join(self.save_dir, "checkpoints"), exist_ok=True)

    def on_train_start(self, trainer):
        print(f"ModelCheckpoint: 检查点将保存到 {self.save_dir}")

    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        current_metric = val_metrics.get(self.monitor, train_metrics.get(self.monitor))

        if current_metric is None:
            print(f"Warning: 监控指标 {self.monitor} 不存在")
            return

        # 定期保存
        if trainer.epoch % self.save_every == 0:
            self._save_checkpoint(trainer, f"epoch_{trainer.epoch}", train_metrics, val_metrics)

        # 保存最佳模型
        if self.save_best:
            is_better = False
            if self.best_metric is None:
                is_better = True
            elif self.mode == "min" and current_metric < self.best_metric:
                is_better = True
            elif self.mode == "max" and current_metric > self.best_metric:
                is_better = True

            if is_better:
                self.best_metric = current_metric
                self.best_epoch = trainer.epoch
                self._save_checkpoint(trainer, f"best_model_{current_metric:.4f}", train_metrics, val_metrics)

                if trainer.accelerator.is_main_process:
                    print(f"🎯 新的最佳模型! {self.monitor}: {current_metric:.4f} (epoch {trainer.epoch})")

    def _save_checkpoint(self, trainer, name: str, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]):
        """保存检查点"""
        checkpoint_path = os.path.join(self.save_dir, "checkpoints", f"{name}.pt")

        # 准备检查点数据
        checkpoint = {
            'epoch': trainer.epoch,
            'global_step': trainer.global_step,
            'model_state_dict': trainer.accelerator.unwrap_model(trainer.model).state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': {**train_metrics, **val_metrics} if hasattr(self, '_last_metrics') else {},
        }

        if trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()

        # 保存检查点
        trainer.accelerator.save(checkpoint, checkpoint_path)

        # 保存训练信息
        info = {
            'epoch': trainer.epoch,
            'global_step': trainer.global_step,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': trainer.config
        }

        with open(os.path.join(self.save_dir, "checkpoints", f"{name}_info.json"), 'w') as f:
            json.dump(info, f, indent=2)


class EarlyStopping(Callback):
    """早停回调"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patience = config["training"].get("early_stopping_patience", 10)
        self.monitor = config["training"].get("monitor_metric", "val_loss")
        self.mode = config["training"].get("monitor_mode", "min")
        self.min_delta = config["training"].get("min_delta", 0.0)

        self.best_metric = None
        self.counter = 0
        self.should_stop = False

    def on_train_start(self, trainer):
        print(f"EarlyStopping: 监控指标 {self.monitor}, 耐心值 {self.patience}")

    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        if self.should_stop:
            return True  # 停止训练

        current_metric = val_metrics.get(self.monitor, train_metrics.get(self.monitor))

        if current_metric is None:
            print(f"Warning: 早停监控指标 {self.monitor} 不存在")
            return False

        # 检查是否改善
        improved = False
        if self.best_metric is None:
            improved = True
        elif self.mode == "min" and current_metric < self.best_metric - self.min_delta:
            improved = True
        elif self.mode == "max" and current_metric > self.best_metric + self.min_delta:
            improved = True

        if improved:
            self.best_metric = current_metric
            self.counter = 0
            if trainer.accelerator.is_main_process:
                print(f"✅ 指标改善: {self.monitor} = {current_metric:.4f}")
        else:
            self.counter += 1
            if trainer.accelerator.is_main_process:
                print(
                    f"⏳ 早停计数: {self.counter}/{self.patience}, {self.monitor} = {current_metric:.4f} (最佳: {self.best_metric:.4f})")

        # 检查是否应该停止
        if self.counter >= self.patience:
            self.should_stop = True
            if trainer.accelerator.is_main_process:
                print(f"🛑 早停触发! 在 epoch {trainer.epoch} 停止训练")
            return True  # 停止训练

        return False


class LearningRateMonitor(Callback):
    """学习率监控回调"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logging_interval = config["training"].get("lr_log_interval", "epoch")  # "epoch" or "step"
        self.step_interval = config["training"].get("lr_log_step_interval", 100)

    def on_batch_end(self, trainer, batch, outputs, loss):
        if self.logging_interval == "step" and trainer.global_step % self.step_interval == 0:
            lr = self._get_current_lr(trainer)
            if lr is not None:
                trainer.accelerator.log({"learning_rate": lr}, step=trainer.global_step)

    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        if self.logging_interval == "epoch":
            lr = self._get_current_lr(trainer)
            if lr is not None:
                trainer.accelerator.log({"learning_rate": lr}, step=trainer.epoch)

                if trainer.accelerator.is_main_process:
                    print(f"📊 学习率: {lr:.2e}")

    def _get_current_lr(self, trainer):
        """获取当前学习率"""
        if trainer.optimizer is None:
            return None

        # 获取第一个参数组的学习率
        for param_group in trainer.optimizer.param_groups:
            return param_group.get('lr', None)

        return None


class ProgressLogger(Callback):
    """训练进度日志回调"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_interval = config["training"].get("log_interval", 50)

    def on_train_start(self, trainer):
        if trainer.accelerator.is_main_process:
            print("🚀 开始训练...")
            print(f"📁 输出目录: {trainer.config['experiment']['output_dir']}")
            print(f"📊 总轮数: {trainer.config['training']['epochs']}")

    def on_epoch_start(self, trainer):
        if trainer.accelerator.is_main_process:
            print(f"\n📅 Epoch {trainer.epoch}/{trainer.config['training']['epochs']}")

    def on_batch_end(self, trainer, batch, outputs, loss):
        if trainer.accelerator.is_main_process and trainer.global_step % self.log_interval == 0:
            print(f"   Step {trainer.global_step}, Loss: {loss.item():.4f}")

    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        if trainer.accelerator.is_main_process:
            metrics_str = []
            train_loss = train_metrics.get("train_loss", 0)
            val_loss = val_metrics.get("val_loss", 0)
            metrics_str.append(f"训练损失: {train_loss:.4f}")
            metrics_str.append(f"验证损失: {val_loss:.4f}")
            print(f"   📈 指标: {', '.join(metrics_str)}")


class MetricsCallback(Callback):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config  # 单独保存配置

    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        """在epoch结束时计算各类指标"""
        if "outputs" not in val_metrics or "labels" not in val_metrics:
            return False

        outputs = val_metrics["outputs"]
        labels = val_metrics["labels"]

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        val_metrics["val_accuracy"] = correct / len(labels)
        print(f"   🧮 计算验证集准确率val_accuracy: {val_metrics['val_accuracy']:.4f}")
        # 计算其他指标...

        # 记录所有指标
        if trainer.accelerator.is_main_process:
            trainer.accelerator.log(val_metrics)

        return False


# 回调函数注册表
def get_callback(name: str):
    callback_registry = {
        "ModelCheckpoint": ModelCheckpoint,
        "EarlyStopping": EarlyStopping,
        "LearningRateMonitor": LearningRateMonitor,
        "ProgressLogger": ProgressLogger,
        "MetricsCallback": MetricsCallback,
    }


    if name not in callback_registry:
        raise ValueError(f"Unknown callback: {name}. Available: {list(callback_registry.keys())}")

    return callback_registry[name]
