import torch


class MseLatent(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(MseLatent, self).__init__()
        self.reduction = reduction
        raise NotImplementedError

    def forward(self, mu_1, logvar_1, mu_2, logvar_2):
        """
        Mean squared Error loss between two latent spaces
        :param mu_1: torch.Tensor, mean of the first distribution
        :param logvar_1: not used, just for interface reasons
        :param mu_2: torch.Tensor, mean of the second distribution
        :param logvar_2: not used, only for interface reasons
        :return: MSE loss
        """
        raise NotImplementedError
