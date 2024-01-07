import os
import numpy as np
from torch.utils.data import Dataset

from dataloaders.utils import get_ray_data


class COLMAP(Dataset):
    def __init__(self, config, split):
        """Constructor for training data loader
            data_config: arguments for data
        """
        #load colmap data
        self.image_path = config['data'][split]['image_path']
        self.workspace_path = config['data'][split]['sparse_recon_path']
                
        #create an array of rays
        self.rays = {}
        self.rays['ray_o'], self.rays['ray_d'], self.rays['ray_rgb'] = get_ray_data(self.image_path, self.workspace_path)
        
        #size of dataset
        self.size = config['data'][split]['size']
        
    def __len__(self):
        """Length of dataset
        """
        return self.size
        
    def __getitem__(self, idx):
        """Get item from dataset
        """
        #select a random camera
        cam_id = np.random.randint(self.num_cameras)
        
        #select a random pixel in the image
        y, x = np.random.randint(self.resolution[0]), np.random.randint(self.resolution[1])
        
        
        #get pixel color
        rgb = self.get_pixel_color(cam_id, x, y)
        
