import torch

def sampler(rays, num_samples, weights=None, fine=False):
    if fine == False:
        #sample points
        gen_diff = torch.linspace(0., 1., num_samples).unsqueeze(0).unsqueeze(-1) #1xTx1
        t_start = (rays['ray_bds'][...,0]).unsqueeze(-1).unsqueeze(-1) #Rx1x1
        t_diff = (rays['ray_bds'][...,1] - rays['ray_bds'][...,0]).unsqueeze(-1).unsqueeze(-1) #Rx1x1
        t_sampled = t_start + gen_diff * t_diff #RxTx1
        rays_o = rays['ray_o'].unsqueeze(1) #Rx1x3
        rays_d = rays['ray_d'].unsqueeze(1) #Rx1x3
        locs = rays_o + t_sampled * rays_d #RxTx3
        
        #convert directions to same shape as pts
        dirs = rays_d.expand(locs.shape)
        
        return locs, dirs
        
    elif fine == True and weights is None:
        raise ValueError('weights must be provided if fine is True')

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map