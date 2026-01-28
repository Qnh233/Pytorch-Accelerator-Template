import torch.nn as nn
from utils.registry import HEADS

@HEADS.register
class ClassificationHead(nn.Module):
    """
    Standard Classification Head: Global Pooling + Linear
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x expected to be (B, C, H, W)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
