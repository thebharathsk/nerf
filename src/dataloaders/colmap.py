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
        self.rays['ray_o'], self.rays['ray_d'], self.rays['ray_rgb'], self.rays['ray_bds'], self.rays['ray_term'], self.rays['reproj_error'] = get_ray_data(self.image_path, self.workspace_path, self.downscale)
        
        #select images
        self.imgs_list = config['data'][split]['imgs_list']
        if self.imgs_list is not None:
            for key in self.rays.keys():
                self.rays[key] = self.rays[key][self.imgs_list]
        
        #keep track of valid depth pixels
        self.valid_depth = torch.where(self.rays['ray_term'] > 0)
        self.num_valid_depth = len(self.valid_depth[0])
        
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
        batch_color = {}
        batch_color['ray_id'] = torch.tensor([cam_id, y, x])
        batch_color['ray_o'] = self.rays['ray_o'][cam_id, y, x]
        batch_color['ray_d'] = self.rays['ray_d'][cam_id, y, x]
        batch_color['ray_rgb'] = self.rays['ray_rgb'][cam_id, y, x]
        batch_color['ray_bds'] = self.rays['ray_bds'][cam_id, y, x]
        batch_color['ray_term'] = self.rays['ray_term'][cam_id, y, x]
        batch_color['reproj_error'] = self.rays['reproj_error'][cam_id, y, x]
        batch_color['color_data'] = 1
        batch_color['depth_data'] = 0
                
        #gather a valid depth pixel
        #sample random depth pixel
        depth_idx = idx%self.num_valid_depth
        depth_cam_id, depth_y, depth_x = self.valid_depth[0][depth_idx], self.valid_depth[1][depth_idx], self.valid_depth[2][depth_idx]
        
        batch_depth = {}
        batch_depth['ray_id'] = torch.tensor([depth_cam_id, depth_y, depth_x])
        batch_depth['ray_o'] = self.rays['ray_o'][depth_cam_id, depth_y, depth_x]
        batch_depth['ray_d'] = self.rays['ray_d'][depth_cam_id, depth_y, depth_x]
        batch_depth['ray_rgb'] = self.rays['ray_rgb'][depth_cam_id, depth_y, depth_x]
        batch_depth['ray_bds'] = self.rays['ray_bds'][depth_cam_id, depth_y, depth_x]
        batch_depth['ray_term'] = self.rays['ray_term'][depth_cam_id, depth_y, depth_x]
        batch_depth['reproj_error'] = self.rays['reproj_error'][depth_cam_id, depth_y, depth_x]
        batch_depth['color_data'] = 0
        batch_depth['depth_data'] = 1
        
        return batch_color, batch_depth