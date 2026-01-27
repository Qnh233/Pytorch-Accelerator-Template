import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from utils.registry import BACKBONES

@BACKBONES.register
class ResNet(nn.Module):
    """
    ResNet Backbone
    Returns the final feature map (before pooling) or a list of feature maps.
    """
    def __init__(self, depth=50, weights=None, out_indices=(3,), pretrained=True):
        super().__init__()

        # Load torchvision model
        if depth == 18:
            model = resnet18(weights=weights if weights else ("IMAGENET1K_V1" if pretrained else None))
        elif depth == 34:
            model = resnet34(weights=weights if weights else ("IMAGENET1K_V1" if pretrained else None))
        elif depth == 50:
            model = resnet50(weights=weights if weights else ("IMAGENET1K_V1" if pretrained else None))
        elif depth == 101:
            model = resnet101(weights=weights if weights else ("IMAGENET1K_V1" if pretrained else None))
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        # Extract layers
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.out_indices = out_indices

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        x = self.layer1(x) # Stage 1
        if 0 in self.out_indices: outs.append(x)

        x = self.layer2(x) # Stage 2
        if 1 in self.out_indices: outs.append(x)

        x = self.layer3(x) # Stage 3
        if 2 in self.out_indices: outs.append(x)

        x = self.layer4(x) # Stage 4
        if 3 in self.out_indices: outs.append(x)

        # If user wants single output, return list implies multi-scale interface is preserved
        return outs
