import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.utils import save_image
from losses.VGG import VGG
from torch.nn import Module
from typing import Callable, Optional

alpha=20
beta=0.1

device=torch.device( "cuda" if (torch.cuda.is_available()) else 'cpu')
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor


class Style_inclusive_loss(Module):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Style_inclusive_loss, self).__init__()
        self.model =VGG().to(device).eval()
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.calculate_loss(input.requires_grad_(), target)
        
    def calc_content_loss(self, gen_feat,orig_feat):
        #calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
        content_l=torch.mean((gen_feat-orig_feat)**2)
        #print("Content loss: ", content_l)
        return content_l

    def calc_style_loss(self, gen_features,orig_feautes):
        #Calculating the gram matrix for the style and the generated image
        batch_size, channel,height,width = orig_feautes.shape

        G=torch.bmm(gen_features.view(batch_size, channel,height*width),torch.transpose(orig_feautes.view(batch_size, channel,height*width), 1, 2))
        A=torch.bmm(orig_feautes.view(batch_size, channel,height*width),torch.transpose(orig_feautes.view(batch_size, channel,height*width), 1, 2))
            
        #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
        style_l=torch.mean((G-A)**2)
        #print("Style loss: ", style_l)
        return style_l


    def calculate_loss(self, generated_batch, ground_truth):
        style_loss=content_loss=0
        gen_features, style_gen_features = self.model(generated_batch)
        orig_feautes, style_orig_features = self.model(ground_truth)
        
        for gen, orig, gen_style, orig_style in zip(gen_features,orig_feautes, style_gen_features, style_orig_features):
            #extracting the dimensions from the generated image
            content_loss+=self.calc_content_loss(gen,orig)
            #style_loss+=self.calc_style_loss(gen_style,orig_style)
        
        #calculating the total loss of e th epoch
        total_loss=alpha*content_loss + beta*style_loss
        #print("total loss", total_loss)
        return total_loss