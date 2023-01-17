# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# load pretrained models
load_pretrained_models = False
# number of epochs of training
n_epochs = 2000
# size of the batches - small so its accurate and nice, but sloww, but i have gpu :)
batch_size = 16
# name of the dataset
dataset_name = 'C:/Users/ismyn/UNI/SEM5/CV/FaceReconstruction/data'
# adam: learning rate
lr = 0.00008
# adam: decay of first order momentum of gradient
b1 = 0.5
# adam: decay of first order momentum of gradient
b2 = 0.999
# number of cpu threads to use during batch generation
n_cpu = 4
# dimensionality of the latent space
latent_dim = 100
# size of each image dimension
img_size = 128
# size of random mask
mask_size = 64
# number of image channels
channels = 3
# interval between image sampling
sample_interval = 100

patch_h, patch_w = int(mask_size / 2 ** 3), int(mask_size / 2 ** 3)
patch = (1, patch_h, patch_w)