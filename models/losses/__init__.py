import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import LOSSES

# Register standard PyTorch losses
LOSSES.register(nn.CrossEntropyLoss, "CrossEntropyLoss")
LOSSES.register(nn.MSELoss, "MSELoss")
LOSSES.register(nn.L1Loss, "L1Loss")
LOSSES.register(nn.BCEWithLogitsLoss, "BCEWithLogitsLoss")

@LOSSES.register
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

@LOSSES.register
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

@LOSSES.register
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.alpha * ce_loss + self.beta * dice_loss


def create_loss(loss_config: dict):
    """根据配置创建损失函数"""
    loss_class = LOSSES.get(loss_config["name"])
    params = loss_config.get("params", {}) or {}  # 确保params不会是None
    return loss_class(**params)
