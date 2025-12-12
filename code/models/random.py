import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 5 stages with downsampling at the first conv of each stage
        self.stage1 = ConvBlock(3, 32, stride=2)    # 256 → 128
        self.stage2 = ConvBlock(32, 64, stride=0)   # 128 → 64
        self.stage3 = ConvBlock(64, 128, stride=0)  # 64 → 32
        self.stage4 = ConvBlock(128, 256, stride=0) # 32 → 16
        self.stage5 = ConvBlock(256, 512, stride=0) # 16 → 8

        # final stabilization
        self.bn_last = nn.BatchNorm2d(512)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # More expressive classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 200)
        )

    def forward(self, x):
        x = (x - 0.5) * 2.0

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.bn_last(x)

        x = self.gap(x).flatten(1)
        return self.classifier(x)   # raw logits
