import torch.nn as nn
import torch.nn.functional as F

from whoodeo.models.common import bilinear_plus, zero_conv


class ESPCN(nn.Module):
    def __init__(self, upscale_factor=2):
        super(ESPCN, self).__init__()
        C = 3

        self.conv1 = nn.Conv2d(C, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, C * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        zero_conv(self.conv3)

    def forward(self, x):
        residual = F.tanh(self.conv1(x))
        residual = F.tanh(self.conv2(residual))
        residual = self.pixel_shuffle(self.conv3(residual))
        return bilinear_plus(x, residual, 1)
