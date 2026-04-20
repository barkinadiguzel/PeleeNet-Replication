import torch
import torch.nn as nn
from .conv_bn import ConvBN

class TwoWayDense(nn.Module):
    def __init__(self, in_ch, k):
        super().__init__()

        self.b1 = nn.Sequential(
            ConvBN(in_ch, 2*k, k=1, s=1, p=0),
            ConvBN(2*k, k, k=3, s=1, p=1)
        )

        self.b2 = nn.Sequential(
            ConvBN(in_ch, k//2, k=1, s=1, p=0),
            ConvBN(k//2, k//2, k=3, s=1, p=1),
            ConvBN(k//2, k//2, k=3, s=1, p=1)
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        out = torch.cat([y1, y2], dim=1)
        return torch.cat([x, out], dim=1)
