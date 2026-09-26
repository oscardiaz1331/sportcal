"""U-Net with an ImageNet ResNet34 encoder: the network of the line segmentation (lab/hockey/train_lines_seg) and of
the keypoint + line model (`HalfResUNet`). The attribute names are part of the checkpoints: renaming one breaks every
saved state dict."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torchvision.models import ResNet34_Weights, resnet34

# ImageNet normalisation of the encoder (RGB)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True))

    def forward(self, x):
        return self.f(x)


class UNetResNet34(nn.Module):
    """U-Net with an ImageNet ResNet34 encoder, output at the input resolution. `pretrained=False` when the weights come
    from a checkpoint anyway (no download)."""

    def __init__(self, ncls, pretrained=True):
        super().__init__()
        r = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # 64,  1/2
        self.pool = r.maxpool
        self.e1, self.e2, self.e3, self.e4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.d4 = Block(512 + 256, 256)
        self.d3 = Block(256 + 128, 128)
        self.d2 = Block(128 + 64, 64)
        self.d1 = Block(64 + 64, 32)
        self.final = nn.Sequential(Block(32, 16), nn.Conv2d(16, ncls, 1))

    @staticmethod
    def _up(x, skip):
        x = Fn.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], 1)

    def forward(self, x):
        s0 = self.stem(x)            # 1/2
        s1 = self.e1(self.pool(s0))  # 1/4
        s2 = self.e2(s1)             # 1/8
        s3 = self.e3(s2)             # 1/16
        s4 = self.e4(s3)             # 1/32
        d = self.d4(self._up(s4, s3))
        d = self.d3(self._up(d, s2))
        d = self.d2(self._up(d, s1))
        d = self.d1(self._up(d, s0))
        d = Fn.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.final(d)


class HalfResUNet(UNetResNet34):
    """UNetResNet34 whose heatmaps stay at half the input resolution: 74+ channels at full resolution run out of memory
    (6 GB at batch 2 on the CPU, before the backward pass).
    ponytail: reading perfect half-resolution heatmaps back with `heatmap_peaks` already costs ~1 px at 1920; a
    quadratic peak fit or an offset head is the upgrade if that ever dominates."""

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.e1(self.pool(s0))
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        d = self.d3(self._up(self.d4(self._up(s4, s3)), s2))
        return self.final(self.d1(self._up(self.d2(self._up(d, s1)), s0)))
