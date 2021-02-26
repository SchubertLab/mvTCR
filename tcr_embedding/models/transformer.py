import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
	def __init__(self, params, hdim, num_seq_labels):
		"""

		:param params: hyperparameters as dict
		:param hdim: output feature dimension
		:param num_seq_labels: number of aa labels, input dim
		"""
		super(TransformerEncoder, self).__init__()

	def forward(self, tcr_seq, tcr_len):
		pass


class TransformerDecoder(nn.Module):
	def __init__(self, params, hdim, num_seq_labels):
		"""

		:param params: hyperparameters as dict
		:param hdim: input feature dimension
		:param num_seq_labels: number of aa labels, output dim
		"""
		super(TransformerDecoder, self).__init__()

	def forward(self, shared_hidden_state, gt_tcr_seq):
		pass
