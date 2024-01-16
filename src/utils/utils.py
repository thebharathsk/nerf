import os
import torch
import logging
import csv

import matplotlib.pyplot as plt
import torch.nn.functional as F

from datetime import datetime

#function to generate timestamp string
def get_time_string():
    #credit => ChatGPT
    now = datetime.now()

    #get the year, month, day, hour, minute, and second as integers
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    second = str(now.second).zfill(2)

    #combine the integers into a string in the desired format
    formatted_date = year + month + day + '_' + hour + minute + second
    
    return formatted_date

#function to create log files
def create_log_files(root):
    #create .log file for logging
    logging.basicConfig(filename=os.path.join(root, 'exp.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    #create csv file for storing metrics
    with open(os.path.join(root, 'metrics.csv'), 'w') as file:
        metrics_file = csv.writer(file)        
    
    return logger, metrics_file

def sampler_coarse(rays, num_samples):
    #set device
    device = rays['ray_o'].device
    
    #sample points uniformly
    gen_diff = torch.linspace(0., 1., num_samples, device=device).unsqueeze(0).unsqueeze(-1) #1xTx1
    t_start = (rays['ray_bds'][...,0]).unsqueeze(-1).unsqueeze(-1) #Rx1x1
    t_diff = (rays['ray_bds'][...,1] - rays['ray_bds'][...,0]).unsqueeze(-1).unsqueeze(-1) #Rx1x1
    t_sampled = t_start + gen_diff * t_diff #RxTx1
    
    #perturb samples
    # get intervals between samples
    mids = .5 * (t_sampled[...,1:,:] + t_sampled[...,:-1,:]) #Rx(T-1)x1
    upper = torch.cat([mids, t_sampled[...,-1:,:]], 1) #RxTx1
    lower = torch.cat([t_sampled[...,0:1,:], mids], 1) #RxTx1
    
    # stratified samples in those intervals
    t_rand = torch.rand(t_sampled.shape, device=device) #RxTx1
    t_sampled_rand = lower + (upper - lower) * t_rand #RxTx1
    
    #convert samples to locations and directions
    rays_o = rays['ray_o'].unsqueeze(1) #Rx1x3
    rays_d = rays['ray_d'].unsqueeze(1) #Rx1x3
    locs = rays_o + t_sampled_rand * rays_d #RxTx3
    
    #convert directions to same shape as pts
    dirs = rays_d.expand(locs.shape)
        
    return locs, dirs, t_sampled_rand

def sampler_fine(rays, t_coarse, weights, num_samples):
    #set device
    device = rays['ray_o'].device
    
    #create bin centers
    bins = (t_coarse[...,1:,0] + t_coarse[...,:-1,0])/2 #Rx(T-1)
    
    #ignore first and last weight
    weights = weights[...,1:-1] #Rx(T-2)
    
    #compute cdf from weights
    weights = weights + 1e-5 #Rx(T-2)
    pdf = weights / torch.sum(weights, -1, keepdim=True) #Rx(T-2)
    cdf = torch.cumsum(pdf, -1) #Rx(T-2)
    cdf = torch.cat([torch.zeros_like(cdf[...,0:1]), cdf], -1)  #Rx(T-1)
    
    #create a baseline of uniformly sampled points
    u = torch.linspace(0., 1., steps=num_samples).to(device) #N
    u = u.expand(list(cdf.shape[:-1]) + [num_samples]) #RxN
    
    #find the closest bin for each sample
    u = u.contiguous() #RxN
    inds = torch.searchsorted(cdf, u, right=True) #RxN
    below = torch.max(torch.zeros_like(inds-1), inds-1) #RxN
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds) #RxN
    inds_g = torch.stack([below, above], -1)  #RxNx2
    
    #gather the bins and cdfs
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] #RxNx(T-1)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) #RxNx2
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) #RxNx2

    denom = (cdf_g[...,1]-cdf_g[...,0]) #RxN
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom) #RxN
    t = (u-cdf_g[...,0])/denom #RxN
    t_fine = (bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])).unsqueeze(-1) #RxNx1
    
    #combine coarse and fine samples
    t_combined, _ = torch.sort(torch.cat([t_fine, t_coarse], 1), 1) #Rx(T+N)x1
    
    #convert samples to locations and directions
    rays_o = rays['ray_o'].unsqueeze(1) #Rx1x3
    rays_d = rays['ray_d'].unsqueeze(1) #Rx1x3
    locs = rays_o + t_combined * rays_d #Rx(T+N)x3
    
    #convert directions to same shape as pts
    dirs = rays_d.expand(locs.shape) #Rx(T+N)x3
    
    return locs, dirs, t_combined
    
def compute_alpha(sigma, dists):
    return 1-torch.exp(-sigma*dists)

def compute_transmittance(alpha):
    return torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]

def render(rgb, sigma, t_sampled, dirs, add_sigma_noise=False, plot=False):
    #set device
    device = rgb.device
    
    #compute distances between samples
    dists = t_sampled[...,1:,0] - t_sampled[...,:-1,0] #Rx(T-1)
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,0:1].shape).to(device)], -1)  #RxT

    # dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    #adjust sigma
    if add_sigma_noise:
        sigma_noise = torch.randn(sigma.shape, device=device)
        sigma = sigma + sigma_noise
    sigma = F.relu(sigma)
    
    alpha = compute_alpha(sigma[...,0], dists)  #RxT
    transmittance = compute_transmittance(alpha)
    
    if plot:
        exp_dir = '/home/bharathsk/projects/nerf/exps/20240116_002224_drop_2_noisy_sigma/'

        sigma_np = sigma.detach().cpu().numpy()
        alpha_np = alpha.detach().cpu().numpy()
        transmittance_np = transmittance.detach().cpu().numpy()
        t_sampled_np = t_sampled.detach().cpu().numpy()
        
        plt.plot(t_sampled_np[0,:,0], sigma_np[0], 'b^')
        plt.savefig(os.path.join(exp_dir, f'sigma.png'))
        plt.close()
        
        plt.plot(t_sampled_np[0,:,0], alpha_np[0], 'b^')
        plt.savefig(os.path.join(exp_dir, f'alpha.png'))
        plt.close()
        
        plt.plot(t_sampled_np[0,:,0], transmittance_np[0], 'b^')
        plt.savefig(os.path.join(exp_dir, f'transmittance.png'))
        plt.close()
    
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * transmittance #RxT
    rgb_rendered = torch.sum(weights[...,None] * rgb, -2)  #Rx3

    # depth_rendered = torch.sum((weights * locs[...,-1]), -1)+1 #R
    depth_rendered = torch.sum(weights * (t_sampled[...,0]*(dirs[:,-1].unsqueeze(1))), -1) #R
    # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    accumulation_rendered = torch.sum(weights, -1) #R

    # if white_bkgd:
    #     rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_rendered, depth_rendered, accumulation_rendered, weights