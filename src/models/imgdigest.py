import torch.nn as nn

class imgdigest(nn.Module):
    def __init__(self, config):
        """Constructor for imgdigest
            config: arguments for model
        """
        super(imgdigest, self).__init__()
        
        #initialize model
        self.fc = nn.ModuleList()
        
        #add input layer
        self.fc.append(nn.Linear(config['model']['input_dim'], config['model']['hidden_dim']))
        self.fc.append(nn.ReLU())
        
        #add hidden layers
        for _ in range(config['model']['num_hidden_layers']):
            self.fc.append(nn.Linear(config['model']['hidden_dim'], config['model']['hidden_dim']))
            self.fc.append(nn.ReLU())

        #add output layer
        self.fc.append(nn.Linear(config['model']['hidden_dim'], config['model']['output_dim']))
        
        #initialize activation
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass for imgdigest

        config:
            x: 2D coordinates

        Returns:
            Outputs for a batch
        """
        #pass through layers
        for l in self.fc:
            x = l(x)
        
        #pass through activation
        out = self.activation(x)
        
        return out
        