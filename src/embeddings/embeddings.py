import math
import torch
import torch.nn as nn

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, config):
        """Constructor for Sinusoidal Embeddings
            embedding_config: arguments for embeddings
        """
        super().__init__()
        
        #create embeddings
        self.num_freq_loc = config['embeddings']['num_freq_loc']
        self.num_freq_dir = config['embeddings']['num_freq_dir']
        
    def forward(self, samples):
        #get locations and directions
        locs = samples['locs'].unsqueeze(3) #RxTx3x1
        dirs = samples['dirs'].unsqueeze(3) #RxTx3x1
        
        #store device
        device = locs.device
        
        #get frequencies
        freqs_loc = torch.Tensor([2**i for i in range(self.num_freq_loc)]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device) #1x1x1xF_loc
        freqs_dir = torch.Tensor([2**i for i in range(self.num_freq_dir)]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device) #1x1x1xF_dir
        
        #get embeddings
        locs_embeddings_sin = torch.sin(freqs_loc*math.pi*locs) #RxTx3xF_loc
        locs_embeddings_cos = torch.cos(freqs_loc*math.pi*locs) #RxTx3xF_loc
        dirs_embeddings_sin = torch.sin(freqs_dir*math.pi*dirs) #RxTx3xF_dir
        dirs_embeddings_cos = torch.cos(freqs_dir*math.pi*dirs) #RxTx3xF_dir
        
        #aggregate embeddings
        locs_embeddings = torch.cat([locs_embeddings_sin, locs_embeddings_cos], dim=-1) #RxTx3x2F_loc
        locs_embeddings = locs_embeddings.permute(0, 1, 3, 2).contiguous() #RxTx2F_locx3
        locs_embeddings = locs_embeddings.view(locs_embeddings.shape[0], locs_embeddings.shape[1], -1) #RxTx6F_loc
        
        dirs_embeddings = torch.cat([dirs_embeddings_sin, dirs_embeddings_cos], dim=-1) #RxTx2x2F_dir
        dirs_embeddings = dirs_embeddings.permute(0, 1, 3, 2).contiguous() #RxTx2F_dirx2
        dirs_embeddings = dirs_embeddings.view(dirs_embeddings.shape[0], dirs_embeddings.shape[1], -1) #RxTx4F_dir
        
        #create a dictionary of embeddings
        embeddings = {}
        embeddings['locs'] = locs_embeddings
        embeddings['dirs'] = dirs_embeddings
        
        return embeddings