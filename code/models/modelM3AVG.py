import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelM3AVG(nn.Module):
    def __init__(self):
        super(ModelM3AVG, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, bias=False)    
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, 7, bias=False)   
        self.conv2_bn = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 64, 7, bias=False)     
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 80, 7, bias=False)      
        self.conv4_bn = nn.BatchNorm2d(80)
        self.conv5 = nn.Conv2d(80, 96, 7, bias=False)     
        self.conv5_bn = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 112, 7, bias=False)    
        self.conv6_bn = nn.BatchNorm2d(112)
        self.conv7 = nn.Conv2d(112, 128, 7, bias=False)   
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 144, 7, bias=False)   
        self.conv8_bn = nn.BatchNorm2d(144)
        self.conv9 = nn.Conv2d(144, 160, 7, bias=False)   
        self.conv9_bn = nn.BatchNorm2d(160)
        self.conv10 = nn.Conv2d(160, 176, 7, bias=False) 
        self.conv10_bn = nn.BatchNorm2d(176)

        # Global average pooling layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 176 * 1 * 1 = 176
        self.fc1 = nn.Linear(176, 200, bias=False)
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
        conv7 = F.relu(self.conv7_bn(self.conv7(conv6)))
        conv8 = F.relu(self.conv8_bn(self.conv8(conv7)))
        conv9 = F.relu(self.conv9_bn(self.conv9(conv8)))
        conv10 = F.relu(self.conv10_bn(self.conv10(conv9)))
        
        # Global average pooling
        x = self.gap(conv10)            # (batch, 176, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 176)

        logits = self.fc1_bn(self.fc1(x))
        return logits
    
    def forward(self, x):
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)
