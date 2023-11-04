from loss.loss import MSE as MSE
from loss.loss import L1 as L1

def get_loss(loss_config):
    """Get loss function for NeRF task

    config:
        name: name of loss function
    """
    if loss_config['name'] == 'mse':
        return MSE(loss_config)
    elif loss_config['name'] == 'l1':
        return L1(loss_config)
    else:
        raise NotImplementedError(f'Loss function {loss_config["name"]} not implemented')