import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
	def __init__(self, hyperparams, hdim):
		super(TransformerEncoder, self).__init__()

	def forward(self, x):
		return x


class TransformerDecoder(nn.Module):
	def __init__(self, hyperparams, hdim):
		super(TransformerDecoder, self).__init__()

	def forward(self, x):
		return x
