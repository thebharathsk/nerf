from dataloaders.imgloader import imgloader as imgloader
from dataloaders.testloader import testloader as testloader
from dataloaders.rowloader import rowloader as rowloader

from torch.utils.data import DataLoader

def get_dataloader(mode, data_config, hyperparams):
    """Get dataloader
        mode: train, val, or test
        data_config: arguments for data
        hyperparams: hyperparameters for training
    """
    #get dataset name
    dataset_name = data_config['name']

    if dataset_name == 'imgloader':
        dataset = imgloader(data_config)
    elif dataset_name == 'testloader':
        dataset = testloader(data_config)
    elif dataset_name == 'rowloader':
        dataset = rowloader(data_config)
    else:
        raise NotImplementedError(f'Model {data_config["name"]} not implemented')
    
    #check if shuffling is needed
    shuffle = True if mode == 'train' else False
    dataloader = DataLoader(dataset, batch_size=hyperparams['bs'], shuffle=shuffle, num_workers=hyperparams['num_workers'])
    
    return dataloader