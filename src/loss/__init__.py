from loss.loss import MSE as MSE
from loss.loss import L1 as L1

def get_loss(config):
    """Get loss function for NeRF task

    config:
        name: name of loss function
    """
    if config['loss']['name'] == 'mse':
        return MSE(config)
    elif config['loss']['name'] == 'l1':
        return L1(config)
    else:
        raise NotImplementedError(f'Loss function {config["loss"]["name"]} not implemented')