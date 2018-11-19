import torch.nn as nn
import torch

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # self.uconv_1 = nn.Sequential(nn.ConvTranspose2d(3, 64, kernel_size=6, stride=2, padding=2))
        # self.conv_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        # self.uconv_2 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2, padding=2))
        # self.conv_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        # self.conv_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        # self.conv_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        # self.conv_5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        # self.conv_6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        # self.conv_7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        # self.conv_8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        # self.conv_9 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2))

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(3, 64, kernel_size=6, stride=2, padding=2),
            nn.ReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(0.2),
            nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2, padding=2),
            nn.ReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(0.2),
            nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )


    def forward(self, x):

        out = self.generator(x)
        return out





