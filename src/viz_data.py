import argparse
import yaml
import torch

import numpy as np
import cv2 as cv2
import open3d as o3d

from dataloaders import get_dataloader
from utils.utils import sampler_coarse, sampler_fine, render

#main function
def main(config):
    """Main function
    
        Args:
            config: configuration for training
    """
    #initialize dataloaders
    train_dataloader = get_dataloader('train', config)
    
    #get a batch from dataloader
    batch_color, batch_depth = next(iter(train_dataloader))
    
    # #access dataset from the dataloader
    # dataset = train_dataloader.dataset
    
    # #get ray data
    # rays = dataset.rays
    # for k in rays.keys():
    #     print(k, rays[k].shape)
    
    # #get point cloud locations
    # points3d = dataset.points3d
        
    # #visualize camera positions
    # camera_centers = rays['ray_o'][:,0,0,:] # (N, 3)
    # pcd_cam = o3d.geometry.PointCloud()
    # pcd_cam.points = o3d.utility.Vector3dVector(camera_centers)
    
    # #visualize pointcloud
    # pcd_pts = o3d.geometry.PointCloud()
    # pcd_pts.points = o3d.utility.Vector3dVector(points3d)
    
    # #draw a cube of size 2 centered at origin. only color edges. dont color faces
    # cube = o3d.geometry.TriangleMesh.create_box(2, 2, 2)
    # cube.paint_uniform_color([0, 0, 0])
    # cube.compute_vertex_normals()
    # cube.compute_triangle_normals()
    # cube.translate((-1, -1, -1))
    
    # #add coordinate axes to point cloud
    # coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=(0, 0, 0))
    
    # #draw a line segments representing rays from camera centers
    # cam_id = 0
    # x = 959
    # y = 539
    # line = o3d.geometry.LineSet()
    # line.points = o3d.utility.Vector3dVector(np.array([camera_centers[cam_id], camera_centers[cam_id] + rays['ray_bds'][cam_id,y,x,-1]*rays['ray_d'][cam_id,y,x]]))
    # line.lines = o3d.utility.Vector2iVector(np.array([[0,1]]))
    # line.colors = o3d.utility.Vector3dVector(np.array([[1,0,0], [0,1,0]]))
    
    # #draw z=1 plane
    # plane = o3d.geometry.TriangleMesh.create_box(width=4, height=4, depth=0.1)
    # plane.translate((-2, -2, 1))
    # plane.paint_uniform_color([0, 0, 0])
    # plane.compute_vertex_normals()
    # o3d.visualization.draw_geometries([pcd_cam, pcd_pts, line, coord_axes, plane])
    
    
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