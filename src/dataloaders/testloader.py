import cv2 as cv2
import numpy as np

import torch
from torch.utils.data import Dataset

class testloader(Dataset):
    def __init__(self, data_config):
        """Constructor for test dataloader
            data_config: arguments for data
        """
        #load image
        self.image = cv2.imread(data_config['image_path'])
        self.image = self.image.astype('float32')
        
        #resolution of image
        self.resolution = self.image.shape[0:2]
        
        #size of dataset
        self.size = self.resolution[0]*self.resolution[1]
        
    def __len__(self):
        """Length of dataset
        """
        return self.size
        
    def __getitem__(self, idx):
        """Get item from dataset
        """
        #get x, y coordinate from idx
        y = idx // self.resolution[1]
        x = idx - y*self.resolution[1]
        
        #normalize x, y coordinate
        x_norm, y_norm = x/self.resolution[1], y/self.resolution[0]
        coords_t = torch.Tensor([x_norm, y_norm])
        coords_t = (coords_t-0.5)/0.5
        coords_unnorm_t = torch.Tensor([x, y])
        
        #get pixel value
        colors_t = torch.from_numpy(self.image[y,x]/255.0)
        
        #pack data into a batch
        data = {'coords': coords_t.float(), 
                'coords_unnorm': coords_unnorm_t.long(),
                'colors': colors_t.float()}
                
        return data