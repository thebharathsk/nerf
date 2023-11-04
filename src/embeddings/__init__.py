from embeddings.embeddings import SinusoidalEmbeddings as SinusoidalEmbeddings

def get_embeddings(embedding_args):
    """Function to get embeddings
        embedding_args: arguments for embeddings
    """
    #get embeddings
    if embedding_args['type'] == 'sinusoidal':
        embeddings = SinusoidalEmbeddings(embedding_args)
    else:
        raise NotImplementedError(f'Embeddings {embedding_args["name"]} is not implemented')
    
    return embeddings