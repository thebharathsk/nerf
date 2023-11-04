import torch.nn as nn

class imgdigest(nn.Module):
    def __init__(self, model_args):
        """Constructor for imgdigest
            model_args: arguments for model
        """
        super(imgdigest, self).__init__()
        
        #initialize model
        self.fc = nn.ModuleList()
        
        #add input layer
        self.fc.append(nn.Linear(model_args['input_dim'], model_args['hidden_dim']))
        self.fc.append(nn.ReLU())
        
        #add hidden layers
        for _ in range(model_args['num_hidden_layers']):
            self.fc.append(nn.Linear(model_args['hidden_dim'], model_args['hidden_dim']))
            self.fc.append(nn.ReLU())

        #add output layer
        self.fc.append(nn.Linear(model_args['hidden_dim'], model_args['output_dim']))
        self.fc.append(nn.ReLU())
        
        #initialize activation
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass for imgdigest

        Args:
            x: 2D coordinates

        Returns:
            Outputs for a batch
        """
        #pass through layers
        x = self.fc(x)
        
        #pass through activation
        out = self.activation(x)
        
        return out
        