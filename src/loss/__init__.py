from loss.loss import MSE as MSE
from loss.loss import L1 as L1

def get_loss(loss_args):
    """Get loss function for NeRF task

    Args:
        name: name of loss function
    """
    if loss_args['name'] == 'mse':
        return MSE(loss_args)
    elif loss_args['name'] == 'l1':
        return L1(loss_args)
    else:
        raise NotImplementedError(f'Loss function {loss_args["name"]} not implemented')