import argparse
import yaml
import torch
import immatch

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from immatch.utils import plot_matches
from tqdm import tqdm

from dataloaders.utils import get_intrinsics, get_extrinsics, get_points, get_image_paths

def get_matcher():
    args = {
        'class': 'LoFTR',
        'ckpt': '/home/bharathsk/projects/image-matching-toolbox/pretrained/loftr/outdoor_ds.ckpt',
        'imsize': -1,
        'no_match_upscale': False,
        'eval_coarse': False,
        'match_threshold': 0.9
    }
    model = immatch.__dict__[args['class']](args)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)
    
    return matcher

def get_skew_symm(p):
    return np.array([[0, -p[2], p[1]],
                     [p[2], 0, -p[0]],
                     [-p[1], p[0], 0]])

def get_fundamental_matrix(P1, P2, ext1):
    #compute camera center of first camera
    C1 = -np.linalg.inv(ext1[0:3, 0:3])@ext1[0:3, 3:4] #3x1
    C1 = np.vstack((C1, 1)) #4x1
    
    #compute epipole of second camera
    e_ = (P2@C1).flatten() #3
    
    #compute skew symmetric matrix of e_
    e_x = get_skew_symm(e_) #3x3
    
    #compute psuedo inverse of P1
    P1_pinv = np.linalg.pinv(P1) #4x3
    
    #compute fundamental matrix
    F = e_x@P2@P1_pinv #3x3
    
    return F
    
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
    
    #save number of cameras
    num_cameras = len(img_paths)
    
    #set of cameras to match
    step = [-128, -64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64, 128]
    
    #construct camera matrices
    P = np.zeros((num_cameras, 3, 4))
    for i in range(0, num_cameras):
        P[i] = intrinsics@extrinsics[i, 0:3]
    
    #create a dictionary to store all matches
    matches_dict = {}
    
    #test matcher
    for i in range(0, num_cameras-1):
        if i > 0:
            break
        # for s in step:
        #     #check if step is valid
        #     if i+s < 0 or i+s >= len(img_paths):
        #         continue
            
        #     #create 'j' variable for second camera
        #     j = i+s
        for j in tqdm(range(i+1, num_cameras)):
            #get matches
            matches, _, _, _ = matcher(img_paths[i], img_paths[j])
            
            pts1 = matches[:,0:2]
            pts2 = matches[:,2:4]
            
            #filter matches based on epipolar constraint
            #find fundamental matrix
            F = get_fundamental_matrix(P[i], P[j], extrinsics[i]) #3x3
            
            #find epipolar lines for pts1
            pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1)))) #Nx3
            pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0], 1)))) #Nx3
            lns2 = (F@pts1_hom.T).T #Nx3
            
            #compute distance of pts2 from epipolar lines
            dist = np.abs(np.sum(lns2*pts2_hom, axis=1))/np.sqrt(np.sum(lns2[:,0:2]**2, axis=1))
            
            #compute mask
            mask = dist < 0.5
            pts1 = pts1[mask]
            pts2 = pts2[mask]
            # print(dist.shape)
            # print(dist.min(), dist.max())
            # print(np.mean(dist), np.std(dist))
            # print(np.sum(dist < 0.25), np.sum(dist < 0.5), np.sum(dist < 1))
            # import sys; sys.exit()
            
            #iterate over matches
            for (pt1, pt2) in zip(pts1, pts2):
                #get coordinates
                x, y, x_, y_ = pt1[0], pt1[1], pt2[0], pt2[1]

                #get key in dictionary for the point
                key = int(i*h*w + y*w + x)
                
                #if key doesnt exist, create a new entry
                if key not in matches_dict.keys():
                    matches_dict[key] = np.array([y*P[i,2] - P[i,1], P[i,0] - x*P[i,2]])
                else:
                    matches_dict[key] = np.vstack((matches_dict[key], np.array([y_*P[j,2] - P[j,1], P[j,0] - x_*P[j,2]])))
         
    #get 3D coordinates
    points3d = []#np.zeros((num_cameras, h, w, 3))
    
    #iterate through dictionary
    for i, key in enumerate(matches_dict.keys()):
        #get coordinates for the point
        n = key//(h*w)
        y = (key - n*h*w)//w
        x = (key - n*h*w)%w
        
        #estimate 3D coordinate
        A = matches_dict[key]
        
        if A.shape[0] < 30:
            continue
        
        #compute SVD of A
        _, _, Vt = np.linalg.svd(A)
        
        #compute 3D coordinates
        X = Vt[-1]
        X = (X/X[-1]).flatten()
        
        points3d.append(X[:-1])
    
    #convert points to np
    points3d = np.array(points3d)
    
    #identify points with non positive depth
    mask = points3d[:, 2] > 0
    points3d = points3d[mask]
    lower_limit = np.percentile(points3d[:, 2], 2)
    upper_limit = np.percentile(points3d[:, 2], 98)
    mask = (points3d[:, 2] < upper_limit) & (points3d[:, 2] > lower_limit)
    points3d = points3d[mask]
    
    print(points3d.shape)
    print(np.min(points3d[:, 2]), np.max(points3d[:, 2]))

    #save points3d as ply
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    o3d.io.write_point_cloud("points.ply", pcd)
    
    
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