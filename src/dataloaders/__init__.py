from dataloaders.imgloader import imgloader as imgloader
from dataloaders.testloader import testloader as testloader


def get_dataloader(mode, data_args):
    """Get dataloader
        data_args: arguments for data
    """
    if mode == 'train':
        return imgloader(data_args)
    elif mode == 'val' or mode == 'test':
        return testloader(data_args)
    else:
        raise NotImplementedError(f'Model {data_args["name"]} not implemented')