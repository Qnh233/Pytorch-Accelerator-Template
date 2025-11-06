# 日志记录模块
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
import sys

class FlexibleLogger:
    def __init__(self, name: str, level: int = logging.INFO):
        """初始化日志记录器

        Args:
            name: 日志记录器名称
            level: 日志级别 (默认: logging.INFO)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # 默认字段模板
        self.default_fields = {
            'timestamp': lambda: datetime.now().isoformat(),
            'level': '%(levelname)s',
            'message': '%(message)s'
        }

        # 设置控制台处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(self._format_template())
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _format_template(self) -> str:
        """生成格式化模板"""
        return json.dumps({k: v if isinstance(v, str) else str(v())
                           for k, v in self.default_fields.items()})

    def add_field(self, name: str, value: Any, is_dynamic: bool = False):
        """添加自定义字段

        Args:
            name: 字段名
            value: 字段值或可调用对象
            is_dynamic: 是否为动态值 (需要每次调用)
        """
        if is_dynamic and not callable(value):
            raise ValueError("动态字段必须为可调用对象")

        self.default_fields[name] = value if is_dynamic else str(value)
        self._update_formatters()

    def _update_formatters(self):
        """更新所有处理器的格式化器"""
        new_formatter = logging.Formatter(self._format_template())
        for handler in self.logger.handlers:
            handler.setFormatter(new_formatter)

    def log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """记录日志

        Args:
            level: 日志级别
            message: 日志消息
            extra: 额外字段 (会覆盖默认字段)
        """
        extra_data = extra or {}
        self.logger.log(level, message, extra=extra_data)

    # 快捷方法
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.log(logging.DEBUG, message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.log(logging.INFO, message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.log(logging.WARNING, message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.log(logging.ERROR, message, extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.log(logging.CRITICAL, message, extra)


# ... 保留原有导入 ...
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64


class TrainingLogger(FlexibleLogger):
    def __init__(self, name: str, log_dir: str = "./logs", level: int = logging.INFO):
        """训练专用日志记录器

        Args:
            name: 日志名称
            log_dir: 日志目录 (默认: ./logs)
            level: 日志级别
        """
        super().__init__(name, level)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 添加训练专用字段
        self.add_field('epoch', 0)
        self.add_field('batch', 0)
        self.add_field('loss', 0.0)
        self.add_field('accuracy', 0.0)
        self.add_field('lr', 0.0)

        # 初始化训练统计
        self.train_metrics = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def log_training_step(self, epoch: int, batch: int, loss: float, lr: float):
        """记录训练步骤

        Args:
            epoch: 当前epoch
            batch: 当前batch
            loss: 当前损失值
            lr: 当前学习率
        """
        self.default_fields['epoch'] = epoch
        self.default_fields['batch'] = batch
        self.default_fields['loss'] = f"{loss:.4f}"
        self.default_fields['lr'] = f"{lr:.6f}"
        self.info(f"Training progress - Epoch {epoch} Batch {batch}")

    def log_validation(self, metrics: Dict[str, float]):
        """记录验证结果

        Args:
            metrics: 包含验证指标的字典
        """
        # 更新内存中的指标
        for k, v in metrics.items():
            if k in self.train_metrics:
                self.train_metrics[k].append(v)

        # 记录到日志
        log_msg = "Validation results - " + " ".join(
            [f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(log_msg, extra=metrics)

    def log_sample_images(self, images: list, titles: list = None,
                          figsize: tuple = (10, 5)):
        """记录采样图片

        Args:
            images: 图片张量列表 (C,H,W)
            titles: 图片标题列表
            figsize: 图表大小
        """
        plt.figure(figsize=figsize)
        for i, img in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(img.permute(1, 2, 0).cpu().numpy())
            if titles and i < len(titles):
                plt.title(titles[i])
            plt.axis('off')

        # 保存为Base64编码
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        self.info("Sample images logged", extra={'images': img_str})

    def save_training_curves(self):
        """保存训练曲线图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(self.train_metrics['loss'], label='Train')
        ax1.plot(self.train_metrics['val_loss'], label='Validation')
        ax1.set_title('Loss Curve')
        ax1.legend()

        # 准确率曲线
        ax2.plot(self.train_metrics['accuracy'], label='Train')
        ax2.plot(self.train_metrics['val_accuracy'], label='Validation')
        ax2.set_title('Accuracy Curve')
        ax2.legend()

        # 保存图片
        curve_path = os.path.join(self.log_dir, f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(curve_path)
        plt.close()

        self.info(f"Training curves saved to {curve_path}")