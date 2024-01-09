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
        self.downscale = config['data']['downscale']
                
        #create an array of rays
        self.rays = {}
        self.rays['ray_o'], self.rays['ray_d'], self.rays['ray_rgb'], self.rays['ray_bds'] = get_ray_data(self.image_path, self.workspace_path, self.downscale)
        
        #select images
        self.imgs_list = config['data'][split]['imgs_list']
        if self.imgs_list is not None:
            for key in self.rays.keys():
                self.rays[key] = self.rays[key][self.imgs_list]
        
        #some useful variables
        self.num_cameras, self.h, self.w  = self.rays['ray_o'].shape[:3]
        
        #size of dataset
        self.is_partial_data = config['data'][split]['size'] is not None
        self.size = min(config['data'][split]['size'], self.num_cameras*self.h*self.w) if self.is_partial_data else self.num_cameras*self.h*self.w 
        
    def __len__(self):
        """Length of dataset
        """
        return self.size
        
    def __getitem__(self, idx):
        """Get item from dataset
        """
        #if dataset is partial, override idx
        if self.is_partial_data:
            idx = np.random.randint(0, self.num_cameras*self.h*self.w)
        
        #select camera id, y and x
        cam_id = idx//(self.h*self.w)
        y = (idx - cam_id*self.h*self.w)//self.w
        x = (idx - cam_id*self.h*self.w)%self.w
                  
        #gather data
        batch = {}
        batch['ray_id'] = torch.tensor([cam_id, y, x])
        batch['ray_o'] = self.rays['ray_o'][cam_id, y, x]
        batch['ray_d'] = self.rays['ray_d'][cam_id, y, x]
        batch['ray_rgb'] = self.rays['ray_rgb'][cam_id, y, x]
        batch['ray_bds'] = self.rays['ray_bds'][cam_id, y, x]
        
        return batch