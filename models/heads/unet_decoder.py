import torch
import torch.nn as nn
from utils.registry import HEADS

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

@HEADS.register
class UNetDecoderHead(nn.Module):
    """
    UNet Decoder Head.
    Takes a list of features (from encoder) and upsamples them.
    """
    def __init__(self, features=[512, 256, 128, 64], num_classes=1):
        super().__init__()
        self.ups = nn.ModuleList()

        # Features come in reverse order: [Bottleneck, Down4, Down3, ...]
        # Bottleneck channels: features[0]*2

        in_channels = features[0] * 2

        for feature in features:
            self.ups.append(
                nn.ConvTranspose2d(in_channels, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.final_conv = nn.Conv2d(features[-1], num_classes, kernel_size=1)

    def forward(self, features):
        # features: [Skip1, Skip2, Skip3, Bottleneck]
        # We need to process them in reverse
        x = features[-1] # Bottleneck
        skips = features[:-1][::-1] # Reverse the rest

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # ConvTranspose
            skip_connection = skips[idx//2]

            if x.shape != skip_connection.shape:
                # Resize x to match skip_connection
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip) # DoubleConv

        return self.final_conv(x)
