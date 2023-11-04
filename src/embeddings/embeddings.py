import torch
import torch.nn as nn

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, embedding_args):
        """Constructor for Sinusoidal Embeddings
            embedding_args: arguments for embeddings
        """
        super(SinusoidalEmbeddings, self).__init__()
        
        #create embeddings
        self.num_freq = embedding_args['num_freq']
        
    def forward(self, coords):
        """Forward pass for Sinusoidal Embeddings
            coords: coordinates to embed
        """        
        #get frequencies
        freqs = torch.Tensor([2**i for i in range(self.num_freq)]).unsqueeze(0).unsqueeze(0) #1x1xF
        
        #increase dimensionality of coords
        coords = coords.unsqueeze(2) #Bx2x1
        
        #get embeddings
        embeddings_sin = torch.sin(freqs*torch.pi*coords) #Bx2xF
        embeddings_cos = torch.cos(freqs*torch.pi*coords) #Bx2xF
        
        #aggregate embeddings
        embeddings = torch.cat([embeddings_sin, embeddings_cos], dim=2) #Bx2x2F
        embeddings = embeddings.permute(0, 2, 1).contiguous() #Bx2Fx2
        embeddings = embeddings.view(embeddings.shape[0], -1) #Bx4F
        
        return embeddings