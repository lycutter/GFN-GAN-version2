import torch
import torch.nn as nn
from spectral import SpectralNorm

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                                    nn.LeakyReLU(0.2))
        self.conv_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2))
        self.conv_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2))
        self.conv_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2))
        self.conv_5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1))

    def forward(self, x):

        feature_map = []

        out1 = self.conv_1(x)
        feature_map.append(out1)
        out2 = self.conv_2(out1)
        feature_map.append(out2)
        out3 = self.conv_3(out2)
        feature_map.append(out3)
        out4 = self.conv_4(out3)
        feature_map.append(out4)

        out = self.conv_5(out4)

        # out = out4.view(x.size(0), -1)
        # out = self.linear(out)

        return feature_map, out