"""Small PyTorch U-Net used by the experimental stroke segmentation path."""

from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for stroke ML training/inference. "
            "Install it explicitly for your environment; it is intentionally "
            "not installed automatically by this project."
        ) from exc
    return torch, nn, functional


torch, nn, F = require_torch()


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class StrokeUNet(nn.Module):
    """CPU-friendly U-Net for 3-class line/node/background segmentation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3, base_channels: int = 16):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base_channels)
        self.down2 = DoubleConv(base_channels, base_channels * 2)
        self.down3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))

        x = self.up2(x3)
        if x.shape[-2:] != x2.shape[-2:]:
            x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        if x.shape[-2:] != x1.shape[-2:]:
            x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv1(torch.cat([x, x1], dim=1))
        return self.out(x)


def build_stroke_unet(num_classes: int = 3, base_channels: int = 16):
    return StrokeUNet(in_channels=1, num_classes=num_classes, base_channels=base_channels)

