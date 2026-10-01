"""U-Net discriminator with spectral normalization, as in Real-ESRGAN.

The output is a per-pixel logit map, with no sigmoid. The loss is the
ordinary logistic GAN loss on those logits.

The power iteration is local because MPS has no vdot, which is what
torch's spectral_norm uses for the singular-value estimate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_norm_weight(weight, u, training, n_power=1, eps=1e-12):
    """Weight divided by its largest singular value. `u` tracks that direction."""
    matrix = weight.reshape(weight.shape[0], -1)
    u_hat = u
    with torch.no_grad():
        for _ in range(n_power):
            v_hat = F.normalize(torch.mv(matrix.t(), u_hat), dim=0, eps=eps)
            u_hat = F.normalize(torch.mv(matrix, v_hat), dim=0, eps=eps)
        if training:
            u.copy_(u_hat)
    sigma = torch.sum(u_hat * torch.mv(matrix, v_hat))
    return weight / sigma


class SNConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        direction = F.normalize(self.weight.new_empty(out_channels).normal_(0, 1), dim=0)
        self.register_buffer("u", direction)

    def forward(self, x):
        weight = spectral_norm_weight(self.weight, self.u, self.training)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
        )


class UNetDiscriminatorSN(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, features, 3, 1, 1)
        self.conv1 = SNConv2d(features, features * 2, 4, 2, 1, bias=False)
        self.conv2 = SNConv2d(features * 2, features * 4, 4, 2, 1, bias=False)
        self.conv3 = SNConv2d(features * 4, features * 8, 4, 2, 1, bias=False)
        self.conv4 = SNConv2d(features * 8, features * 4, 3, 1, 1, bias=False)
        self.conv5 = SNConv2d(features * 4, features * 2, 3, 1, 1, bias=False)
        self.conv6 = SNConv2d(features * 2, features, 3, 1, 1, bias=False)
        self.conv7 = SNConv2d(features, features, 3, 1, 1, bias=False)
        self.conv8 = SNConv2d(features, features, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(features, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, inplace=True)
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)
        x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, inplace=True)
        x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), 0.2, inplace=True)
        x6 = x6 + x0
        out = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), 0.2, inplace=True)
        return self.conv9(out)


def gan_bce(logits, real):
    target = logits.new_ones(logits.shape) if real else logits.new_zeros(logits.shape)
    return F.binary_cross_entropy_with_logits(logits, target)
