import torch.nn as nn
from .conv_bn import ConvBN

class TransitionLayer(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv = ConvBN(in_ch, in_ch, k=1, s=1, p=0)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(x))
