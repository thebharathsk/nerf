from dataloaders.imgloader import imgloader as imgloader
from dataloaders.testloader import testloader as testloader
from dataloaders.rowloader import rowloader as rowloader
from dataloaders.colmap import COLMAP as colmap

from torch.utils.data import DataLoader

def get_dataloader(split, config):
    """Get dataloader
        split: train, val, or test
        config: arguments for data
        hyperparams: hyperparameters for training
    """
    #get dataset name
    dataset_name = config['data'][split]['name']

    if dataset_name == 'imgloader':
        dataset = imgloader(config, split)
    elif dataset_name == 'testloader':
        dataset = testloader(config, split)
    elif dataset_name == 'rowloader':
        dataset = rowloader(config, split)
    elif dataset_name == 'colmap':
        dataset = colmap(config, split)
    else:
        raise NotImplementedError(f'Model {dataset_name} not implemented')
    
    #check if shuffling is needed
    shuffle = True if split == 'train' else False
    dataloader = DataLoader(dataset, batch_size=config['hyperparams']['num_rays'], shuffle=shuffle, num_workers=config['hyperparams']['num_workers'])
    
    return dataloader