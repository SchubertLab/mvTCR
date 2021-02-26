import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
	def __init__(self, params, hdim, num_seq_labels):
		"""

		:param params: hyperparameters as dict
		:param hdim: output feature dimension
		:param num_seq_labels: number of aa labels, input dim
		"""
		super(CNNEncoder, self).__init__()

	def forward(self, tcr_seq, tcr_len):
		pass


class CNNDecoder(nn.Module):
	def __init__(self, params, hdim, num_seq_labels):
		"""

		:param params: hyperparameters as dict
		:param hdim: input feature dimension
		:param num_seq_labels: number of aa labels, output dim
		"""
		super(CNNDecoder, self).__init__()

	def forward(self, shared_hidden_state, gt_input):
		pass
