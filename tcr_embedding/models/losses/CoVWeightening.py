"""
Method re-implemented from:
 https://github.com/rickgroen/cov-weighting/
 Multi-Loss Weighting with Coefficient of Variations
 Authors: Groenendijk, Karaoglu, Gevers, Mensink
"""

import torch


class IndividualCoV:
    def __init__(self, name, comet=None, device='cpu'):
        """
        Initialize an individual CoV weight tracker
        :param name: Name of the loss for logging purposes
        :param comet: Comet ML experiment for logging purposes
        :param device: pytorch device (e.g. cuda:0, cpu, ...)
        """
        self.name = f'CoV_{name}'
        self.comet = comet
        self.device = device

        # running variables for the mean and mean normalized losses
        self.running_mean_L = torch.tensor(0., requires_grad=False, device=device)
        self.running_mean_l = torch.tensor(0., requires_grad=False, device=device)

        # running variables for the sample and population variances
        self.running_s_l = torch.tensor(0., requires_grad=False, device=device)
        self.running_std = torch.tensor(0., requires_grad=False, device=device)
        self.step = 0

    def get_weight(self, loss_new, is_train):
        """
        Provides the weight Value based on the CoV method
        :param loss_new: the loss of the current step
        :param is_train: boolean indicating whether the model is training or validating
        :return: the CoV weight
        """
        weight = self.calculate_weight()
        # If train, update running stats
        if is_train:
            self.keep_running_stats(loss_new)
            self.report(weight)
            self.step += 1
        return weight

    def calculate_weight(self):
        """
        Calculates the current weight
        :return: weight by the CoV method
        """
        if self.step == 0:
            return torch.tensor(1., requires_grad=False, device=self.device)
        else:
            return self.running_std / (self.running_mean_l + 1e-8)

    def keep_running_stats(self, loss_cur):
        """
        Update the running std and mean by Welford's algorithm
        :param loss_cur: the loss of the current step
        :return: updates self.running_mean, self.running_std
        """
        # detach: the gradient should not be passed through the loss weights over time
        loss_new = loss_cur.clone().detach()

        l_0 = loss_new if self.step == 0 else self.running_mean_L
        loss_norm = loss_new / l_0

        if self.step == 0:
            mean_param = torch.tensor(0., requires_grad=False, device=self.device)
        else:
            mean_param = 1. - 1. / (self.step + 1)

        new_running_mean_l = mean_param * self.running_mean_l + (1-mean_param) * loss_norm
        self.running_s_l += (loss_norm - self.running_mean_l) * (loss_norm - new_running_mean_l)
        self.running_mean_l = new_running_mean_l

        running_variance_l = self.running_s_l / (self.step + 1)
        self.running_std = torch.sqrt(running_variance_l + 1e-8)

        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * loss_new

    def report(self, weight):
        """
        Reports the Weight to comet if Experiment is provided.
        :param weight: The CoV weight of the current step
        :return: None, maybe logs to comet
        """
        if self.comet is None:
            return

        report_dict = {
            'weight': weight,
            'mean_l': self.running_mean_l,
            'mean_L': self.running_mean_L,
            's': self.running_s_l,
            'std': self.running_std,

        }
        self.comet.log_metrics(report_dict, prefix=f'{self.name}_', step=self.step)


class CoVWeighter:
    def __init__(self, comet, device):
        """
        Initializes the Coefficient of Variation Weighting method.
        :param comet: Comet ML Experiment to log the individual losses
        :param device: pytorch device (e.g. cuda:0, cpu, ...)
        """
        self.device = device
        self.cov_tcr = IndividualCoV(name='TCR', comet=comet, device=device)
        self.cov_rna = IndividualCoV(name='scRNA', comet=comet, device=device)

    def get_weights(self, loss_tcr_new, loss_rna_new, is_train=True):
        """
        Calculate the weights for TCR and scRNA losses based on the CoV method.
        :param loss_tcr_new: the TCR loss of the current step
        :param loss_rna_new: the scRNA of the current step
        :param is_train: boolean indicating whether the model is training or validating
        :return: (float, float) the loss
        """
        if is_train:
            weight_tcr = self.cov_tcr.get_weight(loss_tcr_new, is_train)
            weight_rna = self.cov_rna.get_weight(loss_rna_new, is_train)
        else:
            weight_tcr = self.cov_tcr.get_weight(loss_tcr_new, is_train)
            weight_rna = self.cov_rna.get_weight(loss_rna_new, is_train)

            # approximately equal weighting by usual magnitude
            # weight_tcr = torch.tensor(2.5, requires_grad=False, device=self.device)
            # weight_rna = torch.tensor(1., requires_grad=False, device=self.device)

        # normalize the weights to sum 1
        param_normalize = weight_tcr + weight_rna
        weight_tcr /= param_normalize
        weight_rna /= param_normalize
        return weight_tcr, weight_rna
