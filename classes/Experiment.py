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
from config import n_cpu, dataset_name, channels, load_pretrained_models, mask_size, sample_interval, img_size, patch,patchUnet, n_epochs
from config import b1, b2, lr # for optimizers
from tqdm import tqdm
from scipy.stats import linregress
from losses.Style_inclusive_loss import Style_inclusive_loss

from models.basic_2.Discriminator import Discriminator as Discriminator_basic_2
from models.basic_2.Generator import Generator as Generator_basic_2

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# Loss function

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
def save_sample(batches_done, generator, test_dataloader, experiment_name, generate_whole_image, imgs, masked_imgs, gen_parts_train, masks, epoch = "pretrain"):
    #print("Saving to:", f"images/{experiment_name}/{batches_done}.png")
    samples, masked_samples, i, masks = next(iter(test_dataloader))
    masks = Variable(masks.type(Tensor))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()  # Upper-left coordinate of mask
    # Generate inpainted image
    gen_mask = generator(masked_samples, masks)
    filled_samples = masked_samples.clone()
    if generate_whole_image:
        # only if unet
        filled_samples = gen_mask
    else:
        filled_samples[:, :, i : i + mask_size, i : i + mask_size] = gen_mask
    # Save sample test
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, f"images/{experiment_name}/{epoch}_{batches_done}.png", nrow=4, normalize=True)
    
    # save sample train
    sample_train = torch.cat((masked_imgs.data, gen_parts_train.data, imgs.data), -2)
    save_image(sample_train, f"images/{experiment_name}/{epoch}_{batches_done}_train.png", nrow=4, normalize=True)

default_transforms =  [
    transforms.Resize((img_size, img_size), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

class Experiment(object):
    def __init__(self, name = 'Basic', 
                 batch_size = 8, 
                 generator_models = [Generator_basic_2],
                 discriminator_models = [Discriminator_basic_2],
                 optimizer = 'adam', overfittingStudy = False,
                 pretrain = True, epochs = n_epochs,
                 dropout_generator = 0,
                 dropout_discriminator = 0, 
                 generate_whole_image = False,
                 style_loss = False
                 ):
            
        self.name = name
        self.transforms_ = default_transforms
        self.batch_size = batch_size 
        self.n_cpu = 1
        self.generator_models = generator_models
        self.discriminator_models = discriminator_models
        self.dropout_generator = dropout_generator
        self.dropout_discriminator = dropout_discriminator
        self.pretrain = pretrain
        self._generator = None
        self._discriminator = None
        self._lossest = None
        self._experiment_losses = None
        self._optimizer_d = None
        self._optimizer_g = None
        self.epochs = epochs
        self.patch = None
        self.generate_whole_image = generate_whole_image
        self._Losses = None
        self.style_loss = style_loss
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
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )
        self.test_dataloader_pretrain = DataLoader(
            ImageDataset(dataset_name, transforms_=self.transforms_, mode = "pretrain", overfittingStudy=overfittingStudy),
            batch_size=self.batch_size,
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
        #gen_adv_losses, gen_content_losses, disc_losses, counter
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
        
    @Losses.setter
    def Losses(self, Losses):
        self._Losses = Losses
        
    @property
    def Losses(self):
        return self._Losses
    
        
    def get_models(self):
        pass
    
    def get_loss(self):
        adversarial_loss = torch.nn.MSELoss()
        pixelwise_loss = torch.nn.L1Loss()
        if self.style_loss:
            contentwise_loss = Style_inclusive_loss()
        else:
            contentwise_loss = pixelwise_loss
        self.Losses = [adversarial_loss, contentwise_loss]
        return adversarial_loss, contentwise_loss
    
    def get_optimizers(self):
        pass
    
    def get_dataloaders(self) -> typing.Tuple[DataLoader, DataLoader]:
        return self.dataloader, self.test_dataloader
    
    def print_plots(self):
        gen_adv_losses, gen_content_losses, disc_losses, counter = self.ExperimentLosses
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=counter, y=gen_adv_losses, mode='lines', name='Gen Adv Loss'))

        fig.update_layout(
            width=1000,
            height=500,
            title=self.name + ":Generator Adversarial Loss",
            xaxis_title="Number of training examples seen",
            yaxis_title="Gen Adversarial Loss (MSELoss)"),
        fig.write_image(f"plots/{self.name}/Generator_Adversarial_loss.png")
        fig.show()
        
        

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=counter, y=gen_content_losses, mode='lines', name='Gen Content Loss', marker_color='orange'))

        fig.update_layout(
            width=1000,
            height=500,
            title=self.name + ": Generator Content Loss",
            xaxis_title="Number of training examples seen",
            yaxis_title="Gen Content Loss (L1 Loss)"),
        fig.write_image(f"plots/{self.name}/Generator_Content_loss.png")
        fig.show()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=counter, y=disc_losses, mode='lines', name='Discriminator Loss', marker_color='seagreen'))

        fig.update_layout(
            width=1000,
            height=500,
            title=self.name + ":Discriminator Loss",
            xaxis_title="Number of training examples seen",
            yaxis_title="Disc Loss (MSELoss)"),
        fig.write_image(f"plots/{self.name}/DiscMSE_loss.png")
        fig.show()
    
    def get_Generator(self, channels = channels):
        self.Generator = self.generator_models[0](channels=channels,out_channels = channels, dropout = self.dropout_generator)
        self.patch = (1, *self.Generator.output_shape)
        
        return self.Generator

    def get_Discriminator(self, channels = channels):
        self.Discriminator = self.discriminator_models[0](channels=channels, dropout = self.dropout_discriminator)
        return self.Discriminator
    
    def set_up(self):
        self.get_loss()
        self.get_Generator()
        self.get_Discriminator() 
        
        torch.cuda.empty_cache()
        
        if load_pretrained_models:
            self.Generator.load_state_dict(torch.load(f"saved_models/Vanilla_AOT/generator0.pth"))
            self.Discriminator.load_state_dict(torch.load(f"saved_models/Vanilla_AOT/discriminator0.pth"))
            print("Using pre-trained Context-Encoder GAN model!")
        
        if cuda:
            self.Generator.cuda()
            self.Discriminator.cuda()
            self.Losses = [loss.cuda() for loss in self.Losses]
            
        self.ExperimentLosses = [],[],[],[]
        
        if self._optimizer_type == 'adam':
            self.Optimizer_G = torch.optim.Adam(self.Generator.parameters(), lr=lr, betas=(b1, b2))
            self.Optimizer_D = torch.optim.Adam(self.Discriminator.parameters(), lr=lr, betas=(b1, b2))
        else:
            self.Optimizer_G = torch.optim.Adam(self.Generator.parameters(), lr=lr, betas=(b1, b2))
            self.Optimizer_D = torch.optim.Adam(self.Discriminator.parameters(), lr=lr, betas=(b1, b2))
            
            
    def balanceD_and_G(self, batch_scors_G, batch_scores_D):
        '''
        Returns boolean values train_G and train_D
        '''
        if len(batch_scors_G) < 16:
            return True, True
        slope_scope = 15
        discriminator_slope, _, _, _, _ = linregress(range(slope_scope), batch_scores_D[-slope_scope:])
        generator_slope, _, _, _, _ = linregress(range(slope_scope), batch_scors_G[-slope_scope:])
        if generator_slope > 0 or discriminator_slope < 0:
            #print("Train generator and not discriminator")
            return True, False
        if discriminator_slope > 0:
            #print("Train generator and discriminator")
            return True, True
        else:
            return True, True
    
    
    def pretrainGenerator(self, epochs= 10, run = None):
        print('Pretraining')
        gen_content_loss = 0
        _, contentwise_loss = self.Losses
        tqdm_bar = tqdm(self.test_dataloader_pretrain, desc=f'Pretrain ', total=int(len(self.test_dataloader_pretrain))) # pretrain on 200 random batches, not to go overboard
        for i, (imgs, masked_imgs, masked_parts, masks) in enumerate(tqdm_bar):
            
            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            masked_parts = Variable(masked_parts.type(Tensor))
            masks = Variable(masks.type(Tensor))
            ## Train Generator ##
            self.Optimizer_G.zero_grad()
            # Generate a batch of images
            gen_parts = self.Generator(masked_imgs, masks)
            if self.generate_whole_image:
                #g_adv = adversarial_loss(Variable(self.Discriminator(gen_parts).type(Tensor)), valid)
                g_content = contentwise_loss(gen_parts, imgs) # normally notimgs but  masked_paarts but 
            else:
                g_content = contentwise_loss(gen_parts, masked_parts)
                
            g_content.backward()
            self.Optimizer_G.step()
            if run is not None:
                run["pretrain/gen_content_loss"].append(g_content)
            if i%100 == 0:
                save_sample("pretrain-x"+ str(i), self.Generator, self.test_dataloader, self.name, self.generate_whole_image, imgs, masked_imgs, gen_parts,masks )
    print('End of pretraining')
            

    def trainEpoch(self, epoch, run):
        adversarial_loss, contentwise_loss = self.Losses
        gen_adv_loss, gen_content_loss, disc_loss = 0, 0, 0
        batch_adv_scores_G, batch_adv_scores_D = [], [] 
        tqdm_bar = tqdm(self.dataloader, desc=f'Training Epoch {epoch} ', total=int(len(self.dataloader)))
        gen_adv_losses, gen_content_losses, disc_losses, counter = self.ExperimentLosses
        for i, (imgs, masked_imgs, masked_parts, masks) in enumerate(tqdm_bar):
            
            
            train_G, train_D = self.balanceD_and_G(batch_adv_scores_G, batch_adv_scores_D)
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], *self.patch).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], *self.patch).fill_(0.0), requires_grad=False)


            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            masked_parts = Variable(masked_parts.type(Tensor))
            masks = Variable(masks.type(Tensor))

            ## Train Generator ##
            if train_G:
                self.Optimizer_G.zero_grad()

            # Generate a batch of images
            gen_parts = self.Generator(masked_imgs, masks)

            # Adversarial and pixelwise loss
            #disc_output = self.Discriminator(gen_parts)
            #print(self.Discriminator(gen_parts).type(Tensor).shape,disc_output.shape, valid.shape)
            #print("Generated parts: ", gen_parts.shape)
            g_adv = adversarial_loss(self.Discriminator(gen_parts).type(Tensor), valid)
            
            if self.generate_whole_image:
                g_content = self.contentwise_loss(gen_parts, imgs) # normally notimgs but  masked_paarts but 
            else:
                
                g_content = self.contentwise_loss(gen_parts, masked_parts)
            # Total loss
            g_loss = 0.01 * g_adv + 0.99 * g_content

            g_loss.backward()
            self.Optimizer_G.step()

            ## Train Discriminator ##
            if train_D:
                self.Optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            if self.generate_whole_image:
                # only do imgs if unet discriminator, else do masked_parts
                real_loss = adversarial_loss(self.Discriminator(imgs), valid)
            else:
                real_loss = adversarial_loss(self.Discriminator(masked_parts), valid)
                
            fake_loss = adversarial_loss(self.Discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            self.Optimizer_D.step()
            
            gen_adv_loss += g_adv.item()
            gen_content_loss += g_content.item()
            disc_loss += d_loss.item()
            
            batch_adv_scores_G.append(g_adv.item())
            batch_adv_scores_D.append(d_loss.item())
            
            run["train/batch_disc_losses"].append(d_loss.item())
            run["train/batch_gen_content_losses"].append(g_content.item())
            run["train/batch_gen_adv_losses"].append(g_adv.item())
            
            tqdm_bar.set_postfix(gen_adv_loss=gen_adv_loss/(i+1), gen_content_loss=gen_content_loss/(i+1), disc_loss=disc_loss/(i+1))
            
            
            # Generate sample at sample interval
            batches_done = epoch * len(self.dataloader) + i
            if batches_done % sample_interval == 0:
                save_sample(batches_done, self.Generator, self.test_dataloader, self.name, self.generate_whole_image,imgs, masked_imgs, gen_parts, masks, epoch )
                
                if batches_done % (sample_interval*15) == 0:
                    torch.save(self.Generator.state_dict(), f"saved_models/{self.name}/generator{epoch}_{batches_done}.pth",  _use_new_zipfile_serialization=False)
                    torch.save(self.Discriminator.state_dict(), f"saved_models/{self.name}/discriminator{epoch}_{batches_done}.pth",  _use_new_zipfile_serialization=False)
        disc_losses.append(disc_loss)
        gen_adv_losses.append(gen_adv_loss)
        gen_content_losses.append(gen_content_loss)
        counter.append(i*self.batch_size + imgs.size(0) + epoch*len(self.dataloader.dataset))
        self.ExperimentLosses = gen_adv_losses, gen_content_losses, disc_losses, counter
            
        return gen_adv_losses, gen_content_losses, disc_losses
        
            

#TODO: ajust lr