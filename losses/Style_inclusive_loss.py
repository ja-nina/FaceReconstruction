import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from losses.VGG import VGG
from torch import Module, _Loss
from typing import Callable, Optional
from torch import Tensor
alpha=1
beta=1

device=torch.device( "cuda" if (torch.cude.is_available()) else 'cpu')

model = VGG()

def calc_content_loss(gen_feat,orig_feat):
    #calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
    content_l=torch.mean((gen_feat-orig_feat)**2)
    return content_l

def calc_style_loss(generated_batch,ground_truth):
    #Calculating the gram matrix for the style and the generated image
    gen_features=model(generated_batch)
    orig_feautes=model(ground_truth)
    
    _, channel,height,width = gen_features.shape

    G=torch.mm(gen_features.view(channel,height*width),orig_feautes.view(channel,height*width).t())
    A=torch.mm(orig_feautes.view(channel,height*width),orig_feautes.view(channel,height*width).t())
        
    #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
    style_l=torch.mean((G-A)**2)
    return style_l


def calculate_loss(gen_features, orig_feautes):
    style_loss=content_loss=0
    for gen,cont in zip(gen_features,orig_feautes):
        #extracting the dimensions from the generated image
        content_loss+=calc_content_loss(gen,cont)
        style_loss+=calc_style_loss(gen,cont)
    
    #calculating the total loss of e th epoch
    total_loss=alpha*content_loss + beta*style_loss 
    return total_loss


class Style_inclusive_loss(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Style_inclusive_loss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.tensor(calculate_loss(input, target), dtype=torch.float)