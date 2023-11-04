from dataloaders.imgloader import imgloader as imgloader
from dataloaders.testloader import testloader as testloader

from torch.utils.data import DataLoader

def get_dataloader(mode, data_config, hyperparams):
    """Get dataloader
        mode: train, val, or test
        data_config: arguments for data
        hyperparams: hyperparameters for training
    """
    if mode == 'train':
        dataset = imgloader(data_config)
    elif mode == 'val' or mode == 'test':
        dataset = testloader(data_config)
    else:
        raise NotImplementedError(f'Model {data_config["name"]} not implemented')
    
    dataloader = DataLoader(dataset, batch_size=hyperparams['bs'], shuffle=True, num_workers=hyperparams['num_workers'])
    
    return dataloader