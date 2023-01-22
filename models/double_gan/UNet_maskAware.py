import torch
from torch import nn
from models.BaseNetwork import BaseNetwork


class UNet_maskAware(BaseNetwork):
    def __init__(self, channels=3, out_channels = 5, dropout= 0):
        super().__init__()
        self.output_shape = (128,128)

        self.conv1 = self.contract_block(channels + 1, 32, 7, 3, dropout=dropout)
        self.conv2 = self.contract_block(32, 64, 3, 1,dropout=dropout)
        self.conv3 = self.contract_block(64, 128, 3, 1, dropout=dropout)
        self.upconv3 = self.expand_block(128, 64, 3, 1, dropout=dropout)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1, dropout=dropout)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1, dropout = dropout)

        self.init_weights()
        
    def forward(self, x, mask):

        # downsampling part
        #print(x, mask)
        
        x = torch.cat((x, mask), dim=1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding, dropout):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding, dropout):

        expand = nn.Sequential(
                            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.Dropout(dropout),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.Dropout(dropout),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand