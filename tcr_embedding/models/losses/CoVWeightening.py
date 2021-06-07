"""
Method re-implemented from:
 https://github.com/rickgroen/cov-weighting/
 Multi-Loss Weighting with Coefficient of Variations
 Authors: Groenendijk, Karaoglu, Gevers, Mensink
"""

import math


class IndividualCoV:
    def __init__(self, name, comet=None):
        """
        Initialize an individual CoV weight tracker
        :param name: Name of the loss for logging purposes
        :param comet: Comet ML experiment for logging purposes
        """
        self.name = f'CoV_weight_{name}'
        self.comet = comet

        self.running_mean_l = 0.
        self.running_mean_L = 0.

        self.running_s_l = 0.
        self.running_std = 0.
        self.step = 0

    def get_weight(self, loss_new):
        """
        Provides the weight Value based on the CoV method
        :param loss_new: the loss of the current step
        :return: the CoV weight
        """
        weight = self.calculate_weight()
        self.keep_running_stats(loss_new)
        self.report_weights(weight)
        self.step += 1
        return weight

    def keep_running_stats(self, loss_new):
        """
        Update the running std and mean by Welford's algorithm
        :param loss_new: the loss of the current step
        :return: updates self.running_mean, self.running_std
        """
        loss_norm = loss_new / self.running_mean_L
        if self.step == 0:
            mean_param = 0.
        else:
            mean_param = 1. - 1. / (self.step + 1)

        new_running_mean_l = mean_param * self.running_mean_l + (1-mean_param) * loss_norm
        self.running_s_l += (loss_norm - self.running_mean_l) * (loss_norm - new_running_mean_l)
        self.running_mean_l = new_running_mean_l

        running_variance_l = self.running_s_l / (self.step + 1)
        self.running_std = math.sqrt(running_variance_l + 1e-8)

        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * loss_new

    def calculate_weight(self):
        """
        Calculates the current weight
        :return: weight by the CoV method
        """
        if self.step == 0:
            return 1.
        else:
            return self.running_std / self.running_mean_l

    def report_weights(self, weight):
        """
        Reports the Weight to comet if Experiment is provided.
        :param weight: The CoV weight of the current step
        :return: None, maybe logs to comet
        """
        if self.comet is None:
            return
        self.comet.log_metric(self.name, weight, step=self.step)


class CoVWeights:
    def __init__(self, comet):
        """
        Initializes the Coefficient of Variation Weighting method.
        :param comet: Comet ML Experiment to log the individual losses
        """
        self.cov_tcr = IndividualCoV(name='TCR', comet=comet)
        self.cov_rna = IndividualCoV(name='scRNA', comet=comet)

    def get_weights(self, loss_tcr_new, loss_rna_new, is_train=True):
        """
        Calculate the weights for TCR and scRNA losses based on the CoV method.
        :param loss_tcr_new: the TCR loss of the current step
        :param loss_rna_new: the scRNA of the current step
        :param is_train: boolean indicating whether the model is training or validating
        :return: (float, float) the loss
        """
        if is_train:
            weight_tcr = self.cov_tcr.get_weight(loss_tcr_new)
            weight_rna = self.cov_rna.get_weight(loss_rna_new)
        else:
            # approximately equal weighting by usual magnitude
            weight_tcr = 2.5
            weight_rna = 1.0

        # normalize the weights to sum 1
        param_normalize = weight_tcr + weight_rna
        weight_tcr /= param_normalize
        weight_rna /= param_normalize
        return weight_tcr, weight_rna
