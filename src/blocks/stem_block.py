import torch
import torch.nn as nn
from .conv_bn import ConvBN

class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBN(3, 32, k=3, s=2, p=1)

        self.left = nn.Sequential(
            ConvBN(32, 16, k=1, s=1, p=0),
            ConvBN(16, 32, k=3, s=2, p=1)
        )

        self.right = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fuse = ConvBN(64, 32, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv1(x)

        l = self.left(x)
        r = self.right(x)

        x = torch.cat([l, r], dim=1)
        return self.fuse(x)
