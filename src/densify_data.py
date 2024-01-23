import argparse
import yaml
import torch
import immatch

import numpy as np
import cv2 as cv2
import open3d as o3d
from immatch.utils import plot_matches

from dataloaders.utils import get_intrinsics, get_extrinsics, get_points, get_image_paths

def get_matcher():
    args = {
        'class': 'LoFTR',
        'ckpt': '/home/bharathsk/projects/image-matching-toolbox/pretrained/loftr/outdoor_ds.ckpt',
        'match_threshold': 0.2,
        'imsize': -1,
        'no_match_upscale': False,
        'eval_coarse': False,
        'match_threshold': 0.5
    }
    model = immatch.__dict__[args['class']](args)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)
    
    return matcher

#main function
def main(config):
    """Main function
    
        Args:
            config: configuration for training
    """
    #LOAD DATA
    #get intrinsics
    intrinsics, (h,w) = get_intrinsics(config['data']['train']['sparse_recon_path'])
    
    #get extrinsics
    extrinsics = get_extrinsics(config['data']['train']['sparse_recon_path'])
    
    #get points
    points, vis, err = get_points(config['data']['train']['sparse_recon_path'])
    
    #get image paths
    img_paths = get_image_paths(config['data']['train']['image_path'], config['data']['train']['sparse_recon_path'])
    
    #LOAD MATCHER
    matcher = get_matcher()
    
    #test matcher
    matches, _, _, _ = matcher(img_paths[0], img_paths[1])
    # plot_matches(img_paths[0], img_paths[1], matches, radius=2, lines=True)
    
     
    
if __name__ == "__main__":
    #set numpy as torch seed
    np.random.seed(0)
    torch.manual_seed(0)
    
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to configuation file')
    args = parser.parse_args()
    
    #load configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    main(config)