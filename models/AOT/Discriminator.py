import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from models.BaseNetwork import BaseNetwork

class Discriminator(BaseNetwork):
    def __init__(self,channels = 3, dropout = 0 ):
        super(Discriminator, self).__init__()
        inc = 3
        self.output_shape = (14,14)
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        #print("disc feature shape", feat.shape)
        return feat