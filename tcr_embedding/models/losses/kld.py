"""
This code was partially adapted from:
Date: 15th March 2021
Availability: https://github.com/theislab/multigrate
"""
import torch


class KLD(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(KLD, self).__init__()
        self.reduction = reduction

    def forward(self, mu, logvar, mu_2=None, logvar_2=None):
        """
        Calculate the Kullbach-Leibler-Divergence between two Gaussians
        :param mu: mean of the first Gaussian
        :param logvar: log(var) of the first Gaussian
        :param mu_2: mean of the second Gaussian (default: 0)
        :param logvar_2: log(var) of the second Gaussian (default: 1)
        :return: loss value
        """
        if mu_2 is None or logvar_2 is None:
            kl = self.univariate_kl_loss(mu, logvar)
        else:
            kl = self.general_kl_loss(mu, logvar, mu_2, logvar_2)
        if self.reduction == 'mean':
            kl = torch.mean(kl)
        elif self.reduction == 'sum':
            kl = torch.sum(kl)
        else:
            raise NotImplementedError(f'reduction method {self.reduction} is not implemented.')
        return kl

    def univariate_kl_loss(self, mu, logvar):
        """
        KL loss between the input and a 0 mean, uni-variance Gaussian
        :param mu: mean of the distribution
        :param logvar: log variance of the distribution
        :return: Kulbach Leibler divergence between distribution and Gaussian
        """
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim=1)
        return kl

    def general_kl_loss(self, mu_1, logvar_1, mu_2, logvar_2):
        """
        KL loss between two distributions
        :param mu_1: mean of the first distribution
        :param logvar_1: log variance of the first distribution
        :param mu_2: mean of the second distribution
        :param logvar_2: log variane of the second distribution
        :return: Kulbach Leibler divergence loss between the two distributions
        """
        kl = logvar_2 - logvar_1 + torch.exp(logvar_1)/torch.exp(logvar_2) + (mu_1-mu_2)**2/torch.exp(logvar_2)-1
        kl = 0.5 * torch.sum(kl)
        return kl
