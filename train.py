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
from models.AOT.Generator import Generator as Generator_AOT
from models.AOT.Discriminator import Discriminator as Discriminator_AOT
from classes.Experiment import Experiment
from config import *

random.seed(42)
import warnings
warnings.filterwarnings("ignore")
from models.basic.Discriminator import Discriminator as BasicDiscriminator
from models.basic.Generator import Generator as BasicGenerator
from models.basic_2.Discriminator import  Discriminator as BasicDiscriminator_2
from models.basic_2.Generator import Generator as BasicGenerator_2
from models.AOT.Discriminator import Discriminator as Discriminator_AOT
from models.AOT.Generator import Generator as BasicGenerator_AOT
from models.double_gan.UNet import UNet as DoubleDiscriminator_UNet
from models.double_gan.UNet_maskAware import UNet_maskAware as DoubleGenerator_UNet
from models.double_gan.Generator import Generator as DoubleGenerator_AOT
import neptune.new as neptune

def train(MyExperiment: Experiment):
    """
    For the purpose of thepresentation the train function requires there to be pre-set experiment ( so that studies can be conducted)
    """
    
    run = neptune.init_run(
    project="zukowskanina1/FaceReconstruction",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkYWNmZjVhMS0xZjQxLTQxNDAtODNiYS1lMmE1ODJlMTk0MjMifQ==",
    ) 
    params = {"name": MyExperiment.name,
            "generator_models": MyExperiment.generator_models[0].__class__.__name__,
            "batch_size": MyExperiment.batch_size,
            "discriminator_models": MyExperiment.discriminator_models,
            "pretrain": MyExperiment.pretrain,
            "generate_whole_image": MyExperiment.generate_whole_image,
            "optimizer": MyExperiment._optimizer_type.__class__.__name__,
            "epochs": MyExperiment.epochs,
            "dropoutGenerator": MyExperiment.dropout_generator,
            "dropoutDiscriminator": MyExperiment.dropout_discriminator}
    run["parameters"] = params

    MyExperiment.set_up()
    if MyExperiment.pretrain:
        MyExperiment.pretrainGenerator(run = run)
        
    for epoch in range(MyExperiment.epochs):
        gen_adv_losses, gen_pixel_losses, disc_losses = MyExperiment.trainEpoch(epoch)
        run["train/disc_losses"].append(disc_losses[-1])
        run["train/gen_pixel_losses"].append(gen_pixel_losses[-1])
        run["train/gen_adv_losses"].append(gen_adv_losses[-1])
        
    run.stop()
        
    
if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    # MyExperiment = Experiment(name = 'TestNeptunAOT',
    #                           generator_models = [BasicGenerator_AOT],
    #                           discriminator_models = [DoubleDiscriminator_UNet],
    #                           generate_whole_image= True,
    #                           overfittingStudy=True,
    #                           pretrain = False,
    #                           dropout_generator = 0.1,
    #                           dropout_discriminator = 0.1,
    #                           epochs = n_epochs)
   # working
    # MyExperiment = Experiment(name = 'BasicOverfitUnet',
    #                           generator_models = [BasicGenerator],
    #                           discriminator_models = [DoubleDiscriminator_UNet],
    #                           generate_whole_image= False,
    #                           overfittingStudy=True,
    #                           pretrain = False,
    #                           dropout_generator = 0.1,
    #                           dropout_discriminator = 0.1,
    #                           epochs = n_epochs)
    # MyExperiment = Experiment(name = 'TestNeptunBasicWithUnet',
    #                           generator_models = [BasicGenerator_2],
    #                           discriminator_models = [DoubleDiscriminator_UNet],
    #                           generate_whole_image= False,
    #                           overfittingStudy=True,
    #                           pretrain = False,
    #                           dropout_generator = 0.1,
    #                           dropout_discriminator = 0.1,
    #                           epochs = n_epochs)
    
    # MyExperiment = Experiment(name = 'TestNeptunBasicWithUnetDouble',
    #                           generator_models = [DoubleGenerator_UNet],
    #                           discriminator_models = [DoubleDiscriminator_UNet],
    #                           generate_whole_image= True,
    #                           overfittingStudy=True,
    #                           pretrain = False,
    #                           dropout_generator = 0.1,
    #                           dropout_discriminator = 0.1,
    #                           epochs = n_epochs)
    MyExperiment = Experiment(name = 'Overfit_Study_DoubleUnet',
                              generator_models = [DoubleGenerator_UNet],
                              discriminator_models = [DoubleDiscriminator_UNet],
                              generate_whole_image= True,
                              overfittingStudy=True,
                              pretrain = False,
                              dropout_generator = 0.01,
                              dropout_discriminator = 0.01,
                              epochs = n_epochs)
    train(MyExperiment)


### losses perceptual loss, R1 regularization, adversarial loss

### conv head, transformer and unet
       
       