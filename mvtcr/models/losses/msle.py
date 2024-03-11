import torch
from torch import nn


class MSLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x_pred, x_true):
        log_pred = torch.log(x_pred+1)
        log_true = torch.log(x_true+1)
        loss = self.mse(log_pred, log_true)
        return loss
