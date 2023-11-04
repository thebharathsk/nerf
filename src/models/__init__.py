from models.imgdigest import imgdigest as imgdigest

def get_model(model_args):
    """Get model

    Args:
        model_args: arguments for model

    Returns:
        model
    """
    if model_args['model'] == 'imgdigest':
        return imgdigest(model_args)
    else:
        raise NotImplementedError(f'Model {model_args["name"]} not implemented')