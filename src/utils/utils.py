import torch

def sampler(rays, num_samples, weights=None, fine=False):
    if fine == False:
        #sample points
        gen_diff = torch.linspace(0., 1., num_samples, device=rays['ray_o'].device).unsqueeze(0).unsqueeze(-1) #1xTx1
        t_start = (rays['ray_bds'][...,0]).unsqueeze(-1).unsqueeze(-1) #Rx1x1
        t_diff = (rays['ray_bds'][...,1] - rays['ray_bds'][...,0]).unsqueeze(-1).unsqueeze(-1) #Rx1x1
        t_sampled = t_start + gen_diff * t_diff #RxTx1
        rays_o = rays['ray_o'].unsqueeze(1) #Rx1x3
        rays_d = rays['ray_d'].unsqueeze(1) #Rx1x3
        locs = rays_o + t_sampled * rays_d #RxTx3
        
        #convert directions to same shape as pts
        dirs = rays_d.expand(locs.shape)
        
        return locs, dirs, t_sampled
    elif fine == True and weights is not None:
        #compute cdf
        weights = weights + 1e-5 #RxT
        pdf = weights / torch.sum(weights, -1, keepdim=True) #RxT
        cdf = torch.cumsum(pdf, -1) #Rx(T-1)
        cdf = torch.cat([torch.zeros_like(cdf[...,0:1]), cdf], -1)  #RxT
        
        #sample from cdf
        
    elif fine == True and weights is None:
        raise ValueError('weights must be provided if fine is True')

def compute_alpha(sigma, dists):
    return 1.-torch.exp(-sigma*dists)

def compute_transmittance(alpha):
    return torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]

def render(rgb, sigma, t_sampled, locs):
    dists = t_sampled[...,1:,0] - t_sampled[...,:-1,0] #Rx(T-1)
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,0:1].shape).to(rgb.device)], -1)  #RxT

    # dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    alpha = compute_alpha(sigma[...,0], dists)  #RxT
    transmittance = compute_transmittance(alpha)
    
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * transmittance #RxT
    rgb_rendered = torch.sum(weights[...,None] * rgb, -2)  #Rx3

    depth_rendered = torch.sum((weights * locs[...,-1]), -1)+1 #Rx1
    # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # acc_map = torch.sum(weights, -1)

    # if white_bkgd:
    #     rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_rendered, depth_rendered