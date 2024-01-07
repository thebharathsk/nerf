from models.imgdigest import imgdigest as imgdigest
from models.nerf import NeRF as NeRF

def get_model(config):
    """Get model

    config:
        config: arguments for model

    Returns:
        model
    """
    if config['model']['name'] == 'imgdigest':
        return imgdigest(config)
    elif config['model']['name'] == 'nerf':
        return NeRF(config)
    else:
        raise NotImplementedError(f'Model {config["model"]["name"]} not implemented')