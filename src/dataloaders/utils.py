import os
import torch
from tqdm import tqdm
import cv2 as cv2
import numpy as np

from dataloaders.colmap_scripts.read_write_model import qvec2rotmat, read_cameras_binary, read_images_binary, read_points3D_binary

def rodrigues_formula(old_dir, new_dir):
    #normalize old_dir and new_dir
    old_dir = old_dir/np.linalg.norm(old_dir)
    new_dir = new_dir/np.linalg.norm(new_dir)
    
    #find rotation matrix
    axis = np.cross(old_dir, new_dir)
    axis = axis/np.linalg.norm(axis)
    cos_theta = np.dot(old_dir, new_dir)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    #skew symmetric matrix of axis
    K = np.array([[0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])

    #rodrigues formula for rotation matrix
    rot_mat = np.eye(3) + sin_theta*K + (1 - cos_theta)*np.matmul(K, K)
    
    return rot_mat

def get_intrinsics(workspace_path):
    #extract data
    cameras = read_cameras_binary(os.path.join(workspace_path, 'cameras.bin'))
    camera_params = cameras[1].params
    h = cameras[1].height
    w = cameras[1].width
    
    #create intrinsics matrix
    intrinsics = np.eye(3)
    intrinsics[0, 0] = camera_params[0]
    intrinsics[1, 1] = camera_params[1]
    intrinsics[0, 2] = camera_params[2]
    intrinsics[1, 2] = camera_params[3]
    
    return intrinsics, (h,w)

def get_extrinsics(workspace_path):
    #extract data
    cameras = read_images_binary(os.path.join(workspace_path, 'images.bin'))
    
    #extract extrinsics as dictionary
    extrinsics_dict = {}
    for c in cameras.values():
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[0:3, 0:3] = qvec2rotmat(c.qvec)
        extrinsic_matrix[0:3, 3] = c.tvec
        extrinsics_dict[c.name] = extrinsic_matrix
    
    #sort keys of dictionary
    keys = list(extrinsics_dict.keys())
    keys.sort()
    extrinsics = np.zeros((len(cameras), 4, 4))
    
    for i, k in enumerate(keys):
        extrinsics[i] = extrinsics_dict[k]
    
    return extrinsics

def get_points(workspace_path):
    #extract image data and map image names to image ids
    cameras = read_images_binary(os.path.join(workspace_path, 'images.bin'))
    
    #extract extrinsics as dictionary
    image_names = []
    image_ids = []
    for c in cameras.values():
        image_names.append(c.name)
        image_ids.append(c.id)
    
    #sort image names
    image_names_argsort = np.argsort(image_names)
    
    #create image id map
    image_id_map = {image_ids[v]:i for i, v in enumerate(image_names_argsort)}
    
    #extract point data
    points3D = read_points3D_binary(os.path.join(workspace_path, 'points3D.bin'))
    
    #extract xyz and visibility flags
    points3D_arr = np.zeros((len(points3D), 3))
    visibility = np.zeros((len(points3D), len(cameras)))
    error = np.zeros((len(points3D)))
    
    for i, point in enumerate(points3D.values()):
        #store point coordinates
        points3D_arr[i] = point.xyz
        
        #store error
        error[i] = point.error
        
        #extract visibility
        for j in point.image_ids:
            visibility[i, image_id_map[j]] = 1
    
    return points3D_arr, visibility, error

def get_image_paths(image_path, workspace_path):
    #extract data
    cameras = read_images_binary(os.path.join(workspace_path, 'images.bin'))
    
    #extract intrinsics as dictionary
    image_names = []
    for c in cameras.values():
        image_names.append(c.name)
    
    #sort keys of dictionary
    keys = list(image_names)
    keys.sort()
    
    #create image paths
    image_paths = [os.path.join(image_path, k) for k in keys]
    
    return image_paths

def get_bounds(pts, min_percentile=0.5, max_percentile=99.5):
    #find lower bounds
    lower_bounds = np.percentile(pts, min_percentile, axis=0)
    
    #find upper bounds
    upper_bounds = np.percentile(pts, max_percentile, axis=0)
    
    return lower_bounds, upper_bounds

def find_camera_motion(extrinsics):
    #find direction of motion of cameras
    motion_dirs = np.zeros((extrinsics.shape[0]-1, 3))
    
    for i in range(1, extrinsics.shape[0]):
        motion_dirs[i-1] = (-np.linalg.inv(extrinsics[i, 0:3, 0:3]))@extrinsics[i, 0:3, 3] - (-np.linalg.inv(extrinsics[i-1, 0:3, 0:3]))@extrinsics[i-1, 0:3, 3]
    
    #find average direction of motion    
    avg_dir = np.mean(motion_dirs, axis=0)
        
    #convert it into a unit vector
    avg_dir /= np.linalg.norm(avg_dir)
    
    return avg_dir
    
def transform_3d(extrinsics, points3d):
    #find direction of motion of cameras
    motion_dir = find_camera_motion(extrinsics)
    
    #find rotation matrix to align motion direction with z-axis
    motion_transformation = np.eye(4)
    motion_transformation[0:3, 0:3] = rodrigues_formula(motion_dir, np.array([0, 0, 1]))
    
    #transform extrinsics and points according to motion transformation
    extrinsics = extrinsics@np.linalg.inv(motion_transformation)
    points3d = motion_transformation @ (np.concatenate([points3d, np.ones((points3d.shape[0], 1))], axis=-1).T)
    points3d = points3d.T
    points3d = points3d[:,0:3]/points3d[:,3:]
        
    #find bounds of points
    lower_bounds, upper_bounds = get_bounds(points3d)
    
    #relax bounds by 20%
    lower_bounds = lower_bounds - 0.15*(upper_bounds - lower_bounds)
    upper_bounds = upper_bounds + 0.15*(upper_bounds - lower_bounds)
    
    #find center of volume
    center = (lower_bounds + upper_bounds) / 2
    
    #find scale factor along each axis
    scale = 2/(upper_bounds - lower_bounds + 1e-6)
    
    #find transform to fit volume into cube of size 2
    cube_transform = np.eye(4)
    cube_transform[0,0] = scale[0]
    cube_transform[1,1] = scale[1]
    cube_transform[2,2] = scale[2]
    cube_transform[0:3, 3] = -center*scale
    
    #transform extrinsics and points according to transform
    extrinsics = extrinsics @ np.linalg.inv(cube_transform)
    points3d = cube_transform @ (np.concatenate([points3d, np.ones((points3d.shape[0], 1))], axis=-1).T)
    points3d = points3d.T
    points3d = points3d[:,0:3]/points3d[:,3:]
    
    #find transformation from original volume to transformed volume
    overall_transformation = cube_transform @ motion_transformation
    
    return extrinsics, points3d, overall_transformation

def get_ray_data(image_path, workspace_path, downscale):
    #get intrinsics, extrinsics, points3d and image paths
    K, (h,w) = get_intrinsics(workspace_path)
    extrinsics = get_extrinsics(workspace_path)
    points3d, visibility, error = get_points(workspace_path)
    image_paths = get_image_paths(image_path, workspace_path)
    
    #adjust scale of intrinsics
    K[0,0] /= downscale
    K[1,1] /= downscale
    K[0,2] /= downscale
    K[1,2] /= downscale
    h = int(h/downscale)
    w = int(w/downscale)
    
    #find transform to fit point cloud volume into cube of size 2
    extrinsics_transformed, points3d_transformed, transform = transform_3d(extrinsics, points3d)
    
    #compute camera to world transformation
    extrinsics_transformed_inv = np.linalg.inv(extrinsics_transformed)

    #filter points outside the cube
    pts_mask = np.all(points3d_transformed >= -1, axis=1) & np.all(points3d_transformed <= 1, axis=1)
    points3d_transformed = points3d_transformed[pts_mask]
    visibility = visibility[pts_mask]
    error = error[pts_mask]

    #number of cameras
    num_cameras = extrinsics.shape[0]
    
    #get ray data
    #create an array containing x,y coordinates
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
    
    #compute ray origins
    camera_centers = -np.linalg.inv(extrinsics_transformed[:, 0:3, 0:3])@extrinsics_transformed[:, 0:3, 3:]
    camera_centers = camera_centers[:,:,0]
    rays_o = np.expand_dims(camera_centers, axis=(1, 2))
    rays_o = np.broadcast_to(rays_o, (num_cameras, h, w, 3)).copy()
    
    #compute ray directions
    rays_d = np.stack([(x-K[0][2])/K[0][0], (y-K[1][2])/K[1][1], np.ones_like(x)], -1) # (h, w, 3)
    rays_d = np.transpose(rays_d, (2, 0, 1)) # (3, h, w)
    rays_d = np.reshape(rays_d, (3, -1)) # (3, h*w)
    rays_d = extrinsics_transformed_inv[:,0:3,0:3]@rays_d #transform rays (N, 3, h*w)
    rays_d = rays_d/np.linalg.norm(rays_d, axis=1, keepdims=True) #normalize ray directions (N, 3, h*w)
    rays_d = np.reshape(rays_d, (num_cameras, 3, h, w)) # (N, 3, h, w)
    rays_d = np.transpose(rays_d, (0,2,3,1)) # (num_cameras, h, w, 3)
    
    #compute min and max t values
    t_min = (-1 - rays_o[...,2:])/rays_d[...,2:]
    t_max = (1 - rays_o[...,2:])/rays_d[...,2:]
    t_min = np.maximum(t_min, 0)
    t_max = np.maximum(t_max, 0)
    
    t_range = np.concatenate([t_min, t_max], axis=-1)
    
    #get rgb data
    print('Loading images...')
    rgb = np.zeros((num_cameras, h, w, 3))
    for i, image_path in tqdm(enumerate(image_paths), total=num_cameras):
        img = cv2.imread(image_path)[:,:,::-1]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        rgb[i] = np.array(img, 'float')/255.0
    
    #get depth data
    print('Creating depth data...')
    z_coord = -np.ones((num_cameras, h, w, 1))
    t_depth = -np.ones((num_cameras, h, w, 1))
    reproj_error = -np.ones((num_cameras, h, w, 1))
    for i in tqdm(range(num_cameras)):
        #get visibility mask for camera
        visibility_mask = visibility[:, i] == 1
        
        #get visible points
        visible_points = points3d_transformed[visibility_mask] # (num_visible_points, 3)
                
        #convert to homogeneous coordinates
        visible_points = np.concatenate([visible_points, np.ones((visible_points.shape[0], 1))], axis=-1) # (num_visible_points, 4)
        
        #project visible points onto image plane
        visible_points_proj = K@extrinsics_transformed[i,0:3]@visible_points.T # (3, num_visible_points)
        visible_points_proj = visible_points_proj/visible_points_proj[2]
        visible_points_proj = visible_points_proj[0:2].T # (num_visible_points, 2)
                
        #round off projected points
        visible_points_proj = np.round(visible_points_proj).astype(np.int32)
        
        #apply limits to projected points
        visible_points_proj[:,0] = np.clip(visible_points_proj[:,0], 0, w-1)
        visible_points_proj[:,1] = np.clip(visible_points_proj[:,1], 0, h-1)
        
        #store depth at 2d points
        z_coord[i, visible_points_proj[:,1], visible_points_proj[:,0]] = visible_points[:,2:3]
        
        #compute t value
        t_depth[i, visible_points_proj[:,1], visible_points_proj[:,0]] = (z_coord[i, visible_points_proj[:,1], visible_points_proj[:,0]] - rays_o[i, visible_points_proj[:,1], visible_points_proj[:,0], 2:])/rays_d[i, visible_points_proj[:,1], visible_points_proj[:,0], 2:]
        
        #store reprpojection error at 2d points
        reproj_error[i, visible_points_proj[:,1], visible_points_proj[:,0],0] = error[visibility_mask]     
        
    #convert into torch tensors
    rays_o = torch.from_numpy(rays_o).contiguous().float()
    rays_d = torch.from_numpy(rays_d).contiguous().float()
    rgb = torch.from_numpy(rgb).contiguous().float()
    t_range = torch.from_numpy(t_range).contiguous().float()
    t_depth = torch.from_numpy(t_depth).contiguous().float()
    reproj_error = torch.from_numpy(reproj_error).contiguous().float()
    
    return rays_o, rays_d, rgb, t_range, t_depth, reproj_error