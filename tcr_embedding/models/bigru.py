import torch
import torch.nn as nn


class BiGRUEncoder(nn.Module):
	def __init__(self, hyperparams, hdim):
		super(BiGRUEncoder, self).__init__()

	def forward(self, x):
		return x


class BiGRUDecoder(nn.Module):
	def __init__(self, hyperparams, hdim):
		super(BiGRUDecoder, self).__init__()

	def forward(self, x):
		return x
