import torch.nn as nn

class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin_relu = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.lin_relu(self.linear(x))