import cv2 as cv2
import numpy as np

import torch
from torch.utils.data import Dataset

class imgloader(Dataset):
    def __init__(self, data_config):
        """Constructor for training data loader
            data_config: arguments for data
        """
        #load image
        self.image = cv2.imread(data_config['image_path'])
        
        #resolution of image
        self.resolution = self.image.shape[0:2]
        
        #size of dataset
        self.size = data_config['size']
        
    def __len__(self):
        """Length of dataset
        """
        return self.size
        
    def __getitem__(self, idx):
        """Get item from dataset
        """
        #randomly sample x, y coordinate
        x, y = np.random.randint(0, self.resolution[0]), \
                np.random.randint(0, self.resolution[1])
        
        #normalize x, y coordinate
        x_norm, y_norm = x/self.resolution[0], y/self.resolution[1]
        coords_t = torch.Tensor([x_norm, y_norm])
        coords_t = 2*coords_t-1
        
        #get pixel value
        colors_t = torch.from_numpy(self.image[x, y]/255.0)
        
        #pack data into a batch
        data = {'coords': coords_t.float(), 
                'colors': colors_t.float()}
        
        return data