import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_layers=50, weights=None):
        super().__init__()
        # 这里简化，实际应根据num_layers选择不同的resnet
        from torchvision.models import resnet50
        self.model = resnet50(weights=weights)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后一层，假设是10类分类
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        # print(f'Input shape: {x.shape}')
        return self.model(x)