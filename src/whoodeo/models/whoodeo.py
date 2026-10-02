import torch.nn as nn


class Whoodeo(nn.Module):
    def __init__(self, upscale_factor=2):
        super(Whoodeo, self).__init__()

        # self.conv1 = nn.Conv2d(3,  96, (7, 7), (1, 1), (3, 3))
        self.conv2 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv3 = nn.Conv2d(64,  32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32,  1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        print(x.shape)
        # x = F.tanh(self.conv1(x))
        # x = F.tanh(self.conv2(x))
        # x = F.tanh(self.conv3(x))
        # x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        return x
