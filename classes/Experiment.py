import typing
import torch
import plotly
import plotly.express as px
import plotly.graph_objects as go
import os 
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
from torch.utils.data import DataLoader
from classes.ImageDataset import ImageDataset
from config import n_cpu, dataset_name, channels, load_pretrained_models, mask_size, sample_interval, img_size, patch
from config import b1, b2, lr # for optimizers
from tqdm import tqdm

from models.basic_2.Discriminator import Discriminator as Discriminator_basic_2
from models.basic_2.Generator import Generator as Generator_basic_2

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
def save_sample(batches_done, generator, test_dataloader, experiment_name):
    samples, masked_samples, i = next(iter(test_dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()  # Upper-left coordinate of mask
    # Generate inpainted image
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i : i + mask_size, i : i + mask_size] = gen_mask
    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, f"images/{experiment_name}/{batches_done}.png", nrow=6, normalize=True)

default_transforms =  [
    transforms.Resize((img_size, img_size), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

class Experiment(object):
    def __init__(self, name = 'Basic', 
                 batch_size = 16, 
                 generator_models = [Generator_basic_2], discriminator_models = [Discriminator_basic_2], optimizer = 'adam', overfittingStudy = False):
        self.name = name
        self.transforms_ = default_transforms
        self.batch_size = batch_size 
        self.n_cpu = 1
        self.generator_models = generator_models
        self.discriminator_models = discriminator_models
        
        self._generator = None
        self._discriminator = None
        self._lossest = None
        self._experiment_losses = None
        self._optimizer_d = None
        self._optimizer_g = None
        
        self._optimizer_type = optimizer
        patch_h, patch_w = int(mask_size / 2 ** 3), int(mask_size / 2 ** 3)
        self.patch = (1, patch_h, patch_w)
        
        self.dataloader = DataLoader(
            ImageDataset(dataset_name, transforms_=self.transforms_, overfittingStudy=overfittingStudy),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_cpu,
        )
        self.test_dataloader = DataLoader(
            ImageDataset(dataset_name, transforms_=self.transforms_, mode="val", overfittingStudy=overfittingStudy),
            batch_size=12,
            shuffle=True,
            num_workers=1,
        )
        
        os.makedirs("images/"+ self.name, exist_ok=True)
        os.makedirs("saved_models/"+ self.name, exist_ok=True)
        os.makedirs("plots/"+ self.name, exist_ok=True)
        
    @property
    def Generator(self):
        return self._generator
    
    @Generator.setter
    def Generator(self, generator):
        self._generator = generator
        
    @property
    def Discriminator(self):
        return self._discriminator
    
    @Discriminator.setter
    def Discriminator(self, discriminator):
        self._discriminator = discriminator
        
    @property
    def Losses(self):
        return self._losses
    
    @Losses.setter
    def Losses(self, losses):
        self._losses = losses
        
    @property
    def ExperimentLosses(self):
        return self._experiment_losses[0]
    
    @ExperimentLosses.setter
    def ExperimentLosses(self, *args):
        #gen_adv_losses, gen_pixel_losses, disc_losses, counter
        self._experiment_losses = args
        
    @property
    def Optimizer_G(self):
        return self._optimizer_g
    
    @Optimizer_G.setter
    def Optimizer_G(self, optimizer_g):
        self._optimizer_g = optimizer_g
        
    @property
    def Optimizer_D(self):
        return self._optimizer_d
    
    @Optimizer_D.setter
    def Optimizer_D(self, optimizer_d):
        self._optimizer_d = optimizer_d
        
        
    def get_models(self):
        pass
    
    def get_loss(self):
        adversarial_loss = torch.nn.MSELoss()
        pixelwise_loss = torch.nn.L1Loss()
        self.Losses = [adversarial_loss, pixelwise_loss]
        return adversarial_loss, pixelwise_loss
    
    def get_optimizers(self):
        pass
    
    def get_dataloaders(self) -> typing.Tuple[DataLoader, DataLoader]:
        return self.dataloader, self.test_dataloader
    
    def print_plots(self):
        gen_adv_losses, gen_pixel_losses, disc_losses, counter = self.ExperimentLosses
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=counter, y=gen_adv_losses, mode='lines', name='Gen Adv Loss'))

        fig.update_layout(
            width=1000,
            height=500,
            title="Generator Adversarial Loss",
            xaxis_title="Number of training examples seen",
            yaxis_title="Gen Adversarial Loss (MSELoss)"),
        fig.write_image(f"plots/{self.name}/Generator_Adversarial_loss.png")
        fig.show()
        
        

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=counter, y=gen_pixel_losses, mode='lines', name='Gen Pixel Loss', marker_color='orange'))

        fig.update_layout(
            width=1000,
            height=500,
            title="Generator Pixel Loss",
            xaxis_title="Number of training examples seen",
            yaxis_title="Gen Pixel Loss (L1 Loss)"),
        fig.write_image(f"plots/{self.name}/Generator_Pixel_loss.png")
        fig.show()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=counter, y=disc_losses, mode='lines', name='Discriminator Loss', marker_color='seagreen'))

        fig.update_layout(
            width=1000,
            height=500,
            title="Discriminator Loss",
            xaxis_title="Number of training examples seen",
            yaxis_title="Disc Loss (MSELoss)"),
        fig.write_image(f"plots/{self.name}/DiscMSE_loss.png")
        fig.show()
    
    def get_Generator(self, channels = channels):
        self.Generator = self.generator_models[0](channels=channels)
        return self.Generator

    def get_Discriminator(self, channels = channels):
        self.Discriminator = self.discriminator_models[0](channels=channels)
        return self.Discriminator
    
    def set_up(self):
        self.get_loss()
        self.get_Generator()
        self.get_Discriminator() 
        
        torch.cuda.empty_cache()
        
        if load_pretrained_models:
            self.Generator.load_state_dict(torch.load(f"saved_models/{self.name}/generator.pth"))
            self.Discriminator.load_state_dict(torch.load(f"saved_models/{self.name}/discriminator.pth"))
            print("Using pre-trained Context-Encoder GAN model!")
        
        if cuda:
            self.Generator.cuda()
            self.Discriminator.cuda()
            self.Losses = [loss.cuda() for loss in self.Losses]
            
        self.Generator.apply(weights_init_normal)
        self.Discriminator.apply(weights_init_normal)
            
        self.ExperimentLosses = [],[],[],[]
        
        if self._optimizer_type == 'adam':
            self.Optimizer_G = torch.optim.Adam(self.Generator.parameters(), lr=lr, betas=(b1, b2))
            self.Optimizer_D = torch.optim.Adam(self.Discriminator.parameters(), lr=lr, betas=(b1, b2))
        else:
            self.Optimizer_G = torch.optim.Adam(self.Generator.parameters(), lr=lr, betas=(b1, b2))
            self.Optimizer_D = torch.optim.Adam(self.Discriminator.parameters(), lr=lr, betas=(b1, b2))
            
        

    def trainEpoch(self, epoch):
        gen_adv_loss, gen_pixel_loss, disc_loss = 0, 0, 0
        tqdm_bar = tqdm(self.dataloader, desc=f'Training Epoch {epoch} ', total=int(len(self.dataloader)))
        for i, (imgs, masked_imgs, masked_parts) in enumerate(tqdm_bar):
            
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            masked_parts = Variable(masked_parts.type(Tensor))

            ## Train Generator ##
            self.Optimizer_G.zero_grad()

            # Generate a batch of images
            gen_parts = self.Generator(masked_imgs)

            # Adversarial and pixelwise loss
            g_adv = adversarial_loss(self.Discriminator(gen_parts), valid)
            g_pixel = pixelwise_loss(gen_parts, masked_parts)
            # Total loss
            g_loss = 0.01 * g_adv + 0.99 * g_pixel

            g_loss.backward()
            self.Optimizer_G.step()

            ## Train Discriminator ##
            self.Optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(self.Discriminator(masked_parts), valid)
            fake_loss = adversarial_loss(self.Discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            self.Optimizer_D.step()
            
            gen_adv_losses, gen_pixel_losses, disc_losses, counter = self.ExperimentLosses
            gen_adv_loss += g_adv.item()
            gen_pixel_loss += g_pixel.item()
            gen_adv_losses.append(g_adv.item())
            gen_pixel_losses.append(g_pixel.item())
            disc_loss += d_loss.item()
            disc_losses.append(d_loss.item())
            counter.append(i*self.batch_size + imgs.size(0) + epoch*len(self.dataloader.dataset))
            
            self.ExperimentLosses = gen_adv_losses, gen_pixel_losses, disc_losses, counter
            tqdm_bar.set_postfix(gen_adv_loss=gen_adv_loss/(i+1), gen_pixel_loss=gen_pixel_loss/(i+1), disc_loss=disc_loss/(i+1))
            
            
            # Generate sample at sample interval
            batches_done = epoch * len(self.dataloader) + i
            if batches_done % sample_interval == 0:
                save_sample(batches_done, self.Generator, self.test_dataloader, self.name)
                
                if batches_done % (sample_interval*3) == 0:
                    torch.save(self.Generator.state_dict(), f"saved_models/{self.name}/generator{epoch}.pth",  _use_new_zipfile_serialization=False)
                    torch.save(self.Discriminator.state_dict(), f"saved_models/{self.name}/discriminator{epoch}.pth",  _use_new_zipfile_serialization=False)
        
            

#TODO: ajust lr
