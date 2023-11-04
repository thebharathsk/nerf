from embeddings.embeddings import SinusoidalEmbeddings as SinusoidalEmbeddings

def get_embeddings(embedding_config):
    """Function to get embeddings
        embedding_config: arguments for embeddings
    """
    #get embeddings
    if embedding_config['name'] == 'sinusoidal':
        embeddings = SinusoidalEmbeddings(embedding_config)
    else:
        raise NotImplementedError(f'Embeddings {embedding_config["name"]} is not implemented')
    
    return embeddings