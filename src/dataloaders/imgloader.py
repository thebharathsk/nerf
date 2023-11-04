import math
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
        self.image = self.image.astype('float32')
        
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
        x_norm, y_norm = np.random.uniform(-1,1), \
                np.random.uniform(-1,1)
        
        #truncate x, y coordinate
        x_norm, y_norm = np.clip(x_norm, -1, 1), np.clip(y_norm, -1, 1)
        
        #get unnormalized x, y coordinate
        x_unnorm, y_unnorm = math.floor(((x_norm+1)/2)*self.resolution[1]), \
                math.floor(((y_norm+1)/2)*self.resolution[0])
                
        #convert data to tensors
        coords_norm_t = torch.Tensor([x_norm, y_norm])
        coords_unnorm_t = torch.Tensor([x_unnorm, y_unnorm])
        
        #get pixel value
        colors_t = torch.from_numpy(self.image[y_unnorm, x_unnorm]/255.0)
        
        #pack data into a batch
        data = {'coords': coords_norm_t.float(),
                'coords_unnorm': coords_unnorm_t.long(),
                'colors': colors_t.float()}
        
        return data