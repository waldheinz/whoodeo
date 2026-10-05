import torch.nn as nn
import torch.nn.functional as F

from whoodeo.models.common import ResidualBlock, bilinear_plus


class ModifiedESPCN(nn.Module):
    """
    A modified ESPCN model incorporating residual blocks for enhanced performance.

    Args:
        scale_factor (int): Upscaling factor (e.g., 2 for 2x upscaling). Default is 2.
        num_filters (int): Number of filters in the convolutional layers. Default is 64.
        num_res_blocks (int): Number of residual blocks. Default is 5.
        in_frames (int): Number of stacked RGB frames in the input. Default is 1.
    """
    def __init__(self, scale_factor=2, num_filters=64, num_res_blocks=5, in_frames=1):
        super(ModifiedESPCN, self).__init__()
        self.in_frames = in_frames
        # Initial convolution to extract features from stacked RGB frames
        self.initial_conv = nn.Conv2d(3 * in_frames, num_filters, kernel_size=3, padding=1)
        # Stack of residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        # Predicts the 2× correction, packed for the pixel shuffle.
        self.final_conv = nn.Conv2d(num_filters, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        residual = F.relu(self.initial_conv(x))
        residual = self.res_blocks(residual)
        residual = self.pixel_shuffle(self.final_conv(residual))
        return bilinear_plus(x, residual, self.in_frames)
