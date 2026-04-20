import torch
import torch.nn as nn
from .conv_bn import ConvBN

class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = ConvBN(in_ch, growth_rate, k=3, s=1, p=1)

    def forward(self, x):
        out = self.conv(torch.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)
