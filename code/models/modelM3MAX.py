import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelM3MAX(nn.Module):
    def __init__(self):
        super(ModelM3MAX, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride = 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 48, 3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(48, 64, 3, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 80, 3, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(80)

        self.conv5 = nn.Conv2d(80, 96, 3, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(96)

        self.conv6 = nn.Conv2d(96, 112, 3, padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(112)

        self.conv7 = nn.Conv2d(112, 128, 3, padding=1, bias=False)
        self.conv7_bn = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 144, 3, padding=1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(144)

        self.conv9 = nn.Conv2d(144, 160, 3, padding=1, bias=False)
        self.conv9_bn = nn.BatchNorm2d(160)

        self.conv10 = nn.Conv2d(160, 176, 3, padding=1, bias=False)
        self.conv10_bn = nn.BatchNorm2d(176)

        # unchanged: 176 * 8 * 8 = 11264
        self.fc1 = nn.Linear(11264, 200, bias=False)
        self.fc1_bn = nn.BatchNorm1d(200)

    def get_logits(self, x):
        x = (x - 0.5) * 2.0

        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv2 = F.max_pool2d(conv2, 2)  # 256 -> 128

        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv4 = F.max_pool2d(conv4, 2)  # 128 -> 64

        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        conv6 = F.relu(self.conv6_bn(self.conv6(conv5)))
        conv6 = F.max_pool2d(conv6, 2)  # 64 -> 32

        conv7 = F.relu(self.conv7_bn(self.conv7(conv6)))
        conv8 = F.relu(self.conv8_bn(self.conv8(conv7)))
        conv8 = F.max_pool2d(conv8, 2)  # 32 -> 16

        conv9 = F.relu(self.conv9_bn(self.conv9(conv8)))
        conv10 = F.relu(self.conv10_bn(self.conv10(conv9)))
        conv10 = F.max_pool2d(conv10, 2)  # 16 -> 8

        # Now conv10 is (batch, 176, 8, 8)
        flat = torch.flatten(conv10.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat))
        return logits

    def forward(self, x):
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)
