import numpy as np
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
    
class ImageDataset(Dataset, ):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train", overfittingStudy = False):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        if overfittingStudy:
            self.files = sorted(glob.glob("%s/00000/000*.png" % root))
            self.files = self.files[:] if mode == "train" else self.files[-12:] # see if overfits
        else:
            self.files = sorted(glob.glob("%s/*/*.png" % root))
            self.files = self.files[:-3000] if mode == "train" else self.files[-3000:] # awful can not be like that - suff
       

    def apply_random_mask(self, img):
        """Randomly masks image"""
        wiggle_room = 16
        #mask =  np.zeros((1, self.img_size, self.img_size), dtype=int)
        y1, x1 = np.random.randint((self.img_size - self.mask_size)//2 - wiggle_room, (self.img_size - self.mask_size)//2 + wiggle_room, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        mask = img[0,:,:].clone()
        mask = mask[None, :]
        mask[:,:,:] = 0
        #print("IMG type: ", img.type)
        masked_img[:, y1:y2, x1:x2] = 1
        mask[0,y1:y2, x1:x2 ] = 1
        #print("Ranodm mask shape", masked_img.shape, "my mask shape: ", mask.shape, "masked part shape: ", masked_part.shape)
        return masked_img, masked_part, mask
    
    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        #mask = np.zeros((1, self.img_size, self.img_size), dtype=int)
        masked_img = img.clone()
        mask = img[0,:,:].clone()
        mask = mask[None, :]
        mask[:,:,:] = 0
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1
        mask[ 0, i : i + self.mask_size, i : i + self.mask_size] = 1
        return masked_img, i, mask

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux, mask = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux, mask = self.apply_center_mask(img)

        return img, masked_img, aux, mask

    def __len__(self):
        return len(self.files)
    
    