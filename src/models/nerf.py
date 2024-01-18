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
                self.intermediate_layers.append(LinearReLU(config['embeddings']['num_freq_loc']*6, config['model']['hidden_dim']))
            else:
                self.intermediate_layers.append(LinearReLU(config['model']['hidden_dim'], config['model']['hidden_dim']))
                
        #final layers
        self.final_layers = nn.ModuleList()
        for i in range(config['model']['num_final_layers']):
            if i == 0:
                self.final_layers.append(LinearReLU(config['model']['hidden_dim']+config['embeddings']['num_freq_loc']*6, config['model']['hidden_dim']))
            else:
                self.final_layers.append(LinearReLU(config['model']['hidden_dim'], config['model']['hidden_dim']))
            
        #volume density output layers
        self.sigma_output = nn.Linear(config['model']['hidden_dim'], 1)
        
        #radiance output layers
        self.features = nn.Linear(config['model']['hidden_dim'], config['model']['hidden_dim'])
        
        self.rgb_output = nn.Sequential(
            LinearReLU(config['model']['hidden_dim']+config['embeddings']['num_freq_dir']*6, config['model']['hidden_dim']//2),
            nn.Linear(config['model']['hidden_dim']//2, 3),
            nn.Sigmoid()
        )
    def forward(self, embeddings):
        x_locs, x_dirs = embeddings['locs'], embeddings['dirs']
        #intermediate layers
        for i, layer in enumerate(self.intermediate_layers):
            if i == 0:
                y = layer(x_locs)
            else:
                y = layer(y)
        
        #append position embeddings
        y = torch.cat([y, x_locs], dim=-1)
        
        #final layers
        for layer in self.final_layers:
            y = layer(y)
        
        #estimate volume density
        sigma = self.sigma_output(y)
        
        #final features
        fts = self.features(y)
        
        #append direction embeddings
        fts = torch.cat([fts, x_dirs], dim=-1)
        
        #estimate rgb
        rgb = self.rgb_output(fts)
        
        #create a dictionary of outputs
        outputs = {}
        outputs['sigma'] = sigma
        outputs['rgb'] = rgb
        
        return outputs