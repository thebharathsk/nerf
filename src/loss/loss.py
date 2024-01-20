import torch
import torch.nn as nn

class MSE(nn.Module):
    def __init__(self, config):
        super(MSE, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, batch, samples, renders):
        #get a mask for color data
        mask = batch['color_data'] == 1
        
        return self.criterion(renders['rgb'][mask], batch['ray_rgb'][mask])
    

class L1(nn.Module):
    def __init__(self, config):
        super(L1, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, batch, samples, renders):
        #get a mask for color data
        mask = batch['color_data'] == 1
        
        return self.criterion(renders['rgb'][mask], batch['ray_rgb'][mask])

class SigmaLoss(nn.Module):
    def __init__(self, config):
        super(SigmaLoss, self).__init__()
        
        #define epsilon
        self.eps = 1e-5
    
    def forward(self, batch, samples, renders):
        #define device
        device = renders['weights'].device
        
        #extract useful variables
        weights = renders['weights']
        t_sampled = samples['t_sampled']
        t_term = batch['ray_term']
        reproj_error = batch['reproj_error']
        
        #compute delta in t
        delta_t = t_sampled[...,1:,0] - t_sampled[...,:-1,0] #Rx(T-1)
        delta_t = torch.cat([delta_t, torch.Tensor([1e10]).expand(delta_t[...,0:1].shape).to(device)], -1)  #RxT
        
        #find valid samples
        mask = reproj_error[...,0] != -1 #R
        
        #if no valid samples, return 0
        if torch.sum(mask) == 0:
            return torch.Tensor([0]).to(device)
        
        t_sampled = t_sampled[...,0][mask] #MxT
        delta_t = delta_t[mask] #MxT
        weights = weights[mask] #MxT
        t_term = t_term[mask] #Mx1
        reproj_error = reproj_error[mask] #Mx1
        
        #compute sigma loss
        sigma_loss = -torch.log(weights + self.eps)*torch.exp((-(t_sampled - t_term)**2)/(2*reproj_error**2))*delta_t #MxT
        sigma_loss = torch.sum(sigma_loss, -1) #M
        sigma_loss = torch.mean(sigma_loss) #1

        return sigma_loss

class DepthLoss(nn.Module):
    def __init__(self, config):
        super(DepthLoss, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, batch, samples, renders):
        #define device
        device = renders['weights'].device
        
        #extract useful variables
        weights = renders['weights'] #RxT
        t_sampled = samples['t_sampled'] #RxTx1
        t_term = batch['ray_term'] #Rx1
        reproj_error = batch['reproj_error'] #Rx1
        
        #find valid samples
        mask = batch['depth_data'] == 1 #R
        
        #if no valid samples, return 0
        if torch.sum(mask) == 0:
            return torch.Tensor([0]).to(device)
        
        t_sampled = t_sampled[...,0][mask] #MxT
        weights = weights[mask] #MxT
        t_term = t_term[mask] #Mx1
        reproj_error = reproj_error[mask] #Mx1
        
        #compute depth loss
        depth_pred = torch.sum(weights * t_sampled, -1) #M
        depth_loss = self.criterion(depth_pred, t_term[...,0]) #1
        
        return depth_loss

class CompositeLoss(nn.Module):
    def __init__(self, loss_list, tag):
        """Constructor for MSE Loss
            loss_config: arguments for loss function
        """
        super(CompositeLoss, self).__init__()
        self.loss_list = loss_list
        self.tag = tag
    
    def forward(self, *args):
        loss = {}
        
        #iterate over all losses
        for (loss_name, loss_fn, loss_weight) in self.loss_list:
            loss[loss_name + '_' + self.tag] = loss_weight*loss_fn(*args)
        
        return loss