import os
import torch
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
        self.rays['ray_o'], self.rays['ray_d'], self.rays['ray_rgb'], self.rays['ray_bds'] = get_ray_data(self.image_path, self.workspace_path)
        
        #some useful variables
        self.num_cameras, self.h, self.w  = self.rays['ray_o'].shape[:3]
        
        #size of dataset
        self.size = config['data'][split]['size'] if config['data'][split]['size'] is not None else self.num_cameras*self.h*self.w 
        
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
        y, x = np.random.randint(self.h), np.random.randint(self.w)
        
        #gather data
        batch = {}
        batch['ray_id'] = torch.tensor([cam_id, y, x])
        batch['ray_o'] = self.rays['ray_o'][cam_id, y, x]
        batch['ray_d'] = self.rays['ray_d'][cam_id, y, x]
        batch['ray_rgb'] = self.rays['ray_rgb'][cam_id, y, x]
        batch['ray_bds'] = self.rays['ray_bds'][cam_id, y, x]
        
        return batch