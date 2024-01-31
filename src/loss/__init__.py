from loss.loss import MSE as MSE
from loss.loss import L1 as L1
from loss.loss import SigmaLoss as SigmaLoss
from loss.loss import DepthLoss as DepthLoss
from loss.loss import SparsityLoss as SparsityLoss
from loss.loss import CompositeLoss

def get_loss(config, tag='coarse'):
    """Get loss function for NeRF task

    config:
        name: name of loss function
    """
    #create a dictionary of loss functions
    loss_fn = {}
    loss_fn['mse'] = MSE
    loss_fn['l1'] = L1
    loss_fn['sigma'] = SigmaLoss
    loss_fn['depth'] = DepthLoss
    loss_fn['sparsity'] = SparsityLoss
    
    #create loss list
    loss_list = []
    for loss_name, loss_weight in zip(config['loss']['names'], config['loss']['weights']):
        if loss_name not in loss_fn.keys():
            raise NotImplementedError(f'Loss function {loss_name} not implemented')
        else:
            loss_list.append([loss_name, loss_fn[loss_name](config), loss_weight])
    
    #create composite loss
    loss_fn = CompositeLoss(loss_list, tag)
    
    return loss_fn