import torch.nn as nn
from torchsummary import summary
from models.BaseNetwork import BaseNetwork
class Discriminator(BaseNetwork):
    def __init__(self, channels=3, dropout = 0):
        super(Discriminator, self).__init__()
        self.output_shape = (8,8)
        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
        
        

        self.model = nn.Sequential(*layers)
    def forward(self, img):
        res = self.model(img)
        return res