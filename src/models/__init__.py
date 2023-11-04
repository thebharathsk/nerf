from models.imgdigest import imgdigest as imgdigest

def get_model(model_config):
    """Get model

    config:
        model_config: arguments for model

    Returns:
        model
    """
    if model_config['name'] == 'imgdigest':
        return imgdigest(model_config)
    else:
        raise NotImplementedError(f'Model {model_config["name"]} not implemented')