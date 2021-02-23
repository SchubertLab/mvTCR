# https://github.com/theislab/multigrate
import torch
from functools import reduce


class KLD(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(KLD, self).__init__()
        self.reduction = reduction

    def forward(self, mu, logvar):
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim=1)
        if self.reduction == 'mean':
            kl = torch.mean(kl)
        elif self.reduction == 'sum':
            kl = torch.sum(kl)
        else:
            raise NotImplementedError(f'reduction method {self.reduction} is not implemented.')

        return kl
