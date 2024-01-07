import torch
import torch.nn as nn
from models.layers import LinearReLU

class NeRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        #intermediate layers
        self.intermediate_layers = nn.ModuleList()
        for i in range(config['model']['num_intermediate_layers']):
            if i == 0:
                self.intermediate_layers.append(LinearReLU(config['embeddings']['num_freq_loc']*3, config['model']['hidden_dim']))
            else:
                self.intermediate_layers.append(LinearReLU(config['model']['hidden_dim'], config['model']['hidden_dim']))
                
        #final layers
        self.final_layers = nn.ModuleList()
        for i in range(config['model']['num_final_layers']):
            if i == 0:
                self.final_layers.append(LinearReLU(config['model']['hidden_dim']+config['embeddings']['num_freq_loc']*3, config['model']['hidden_dim']))
            else:
                self.final_layers.append(LinearReLU(config['model']['hidden_dim'], config['model']['hidden_dim']))
            
        #volume density output layers
        self.sigma_output = LinearReLU(config['model']['hidden_dim'], 1)
        
        #radiance output layers
        self.features = nn.Linear(config['model']['hidden_dim'], config['model']['hidden_dim'])
        
        self.rgb_output = nn.Sequential(
            LinearReLU(config['model']['hidden_dim']+config['embeddings']['num_freq_dir']*3, config['model']['hidden_dim']//2),
            nn.Linear(config['model']['hidden_dim']//2, 3),
            nn.Sigmoid()
        )
    def forward(self, x):
        #get location and direction embeddings
        x_loc, x_dirs = x['loc'], x['dir']
        
        #intermediate layers
        y = self.intermediate_layers(x_loc)
        
        #append position embeddings
        y = torch.cat([y, x_loc], dim=-1)
        
        #final layers
        y = self.final_layers(y)
        
        #estimate volume density
        sigma = self.sigma_output(y)
        
        #final features
        fts = self.features(y)
        
        #append direction embeddings
        fts = torch.cat([fts, x_dirs], dim=-1)
        
        #estimate rgb
        rgb = self.rgb_output(fts)
        
        return sigma, rgb
        