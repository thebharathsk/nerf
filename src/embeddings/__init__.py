from embeddings.embeddings import SinusoidalEmbeddings as SinusoidalEmbeddings

def get_embeddings(config):
    """Function to get embeddings
        config: arguments for embeddings
    """
    #get embeddings
    if config['embeddings']['name'] == 'sinusoidal':
        embeddings = SinusoidalEmbeddings(config)
    else:
        raise NotImplementedError(f'Embeddings {config["embeddings"]["name"]} is not implemented')
    
    return embeddings