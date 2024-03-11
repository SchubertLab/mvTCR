# https://github.com/theislab/multigrate
import torch

class NB(torch.nn.Module):
    """
    Yang Comment: Usage in forward:
    x : Ground truth
    mu: Prediction
    theta: Another trainable parameter with shape=[xdim(number of count variables)],
           simply initialize a nn.Parameter(torch.randn(xdim)) in the model

    Be careful, we need the negative of the returned value: loss = -NBLoss
    """
    def __init__(self, eps=1e-8, reduction='mean'):
        super(NB, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor):

        if theta.ndimension() == 1:
            theta = theta.view(1, theta.size(0))

        log_theta_mu_eps = torch.log(theta + mu + self.eps)
        res = (
            theta * (torch.log(theta + self.eps) - log_theta_mu_eps)
            + x * (torch.log(mu + self.eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )

        if self.reduction == 'mean':
            res = torch.mean(res)
        elif self.reduction == 'sum':
            res = torch.sum(res)
        else:
            raise NotImplementedError(f'reduction method {self.reduction} is not implemented.')

        return res

"""
Code Snippet from vae base
prediction = F.softmax(prediction, dim=-1)
size_factor_view = size_factor.unsqueeze(1).expand(-1, prediction.shape[1]).to(prediction.device)
dec_mean = prediction * size_factor_view
dispersion = self.model.theta.T
dispersion = torch.exp(dispersion)
loss = - scRNA_criterion(prediction, dec_mean,
                                          dispersion) * 1000  # TODO A Test, maybe delete afterwards
"""