import numpy as np
import pandas as pd
import os, math, sys
import glob, itertools
import argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchvision.models import vgg19

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models.basic_2.Generator import Generator
from models.basic_2.Discriminator import Discriminator
from classes.Experiment import Experiment
from config import *

random.seed(42)
import warnings
warnings.filterwarnings("ignore")

def train(MyExperiment: Experiment, epochs = n_epochs):
    """
    For the purpose of thepresentation the train function requires there to be pre-set experiment ( so that studies can be conducted)

    Args:
        MyExperiment (Experiment): _description_
        epochs (_type_, optional): _description_. Defaults to n_epochs.
    """
    
    dataloader, test_dataloader = MyExperiment.get_dataloaders()
    adversarial_loss, pixelwise_loss = MyExperiment.get_loss()
    generator, discriminator = MyExperiment.get_Generator(), MyExperiment.get_Discriminator() 
    MyExperiment.set_up()
    for epoch in range(epochs):
        MyExperiment.trainEpoch(epoch)
        
    
if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    MyExperiment = Experiment(name = 'Basic_see_if_works')
    dataloader, test_dataloader = MyExperiment.get_dataloaders()
    adversarial_loss, pixelwise_loss = MyExperiment.get_loss()
    generator, discriminator = MyExperiment.get_Generator(), MyExperiment.get_Discriminator() 
    MyExperiment.set_up()
    for epoch in range(n_epochs):
        MyExperiment.trainEpoch(epoch)


### losses perceptual loss, R1 regularization, adversarial loss

### conv head, transformer and unet
       
       