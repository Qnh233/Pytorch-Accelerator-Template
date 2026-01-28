import torch.nn as nn
from utils.registry import STEMS

@STEMS.register
class ConvStem(nn.Module):
    """
    Standard Convolutional Stem: Conv2d -> BN -> ReLU
    Maps arbitrary input channels to backbone input channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

@STEMS.register
class IdentityStem(nn.Module):
    """
    Pass-through stem.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
