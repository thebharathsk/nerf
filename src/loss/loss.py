import torch.nn as nn

class MSE(nn.Module):
    def __init__(self, loss_config):
        """Constructor for MSE Loss
            loss_config: arguments for loss function
        """
        super(MSE, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        """Forward pass for MSE loss

        config:
            pred: predicted values
            target: target values

        Returns:
            Loss for a batch
        """
        return self.criterion(pred, target)
    

class L1(nn.Module):
    def __init__(self, loss_config):
        """Constructor for MSE Loss
            loss_config: arguments for loss function
        """
        super(L1, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        """Forward pass for MSE loss

        config:
            pred: predicted values
            target: target values

        Returns:
            Loss for a batch
        """
        return self.criterion(pred, target)