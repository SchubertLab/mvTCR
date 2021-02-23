import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
	def __init__(self, hyperparams, hdim):
		super(CNNEncoder, self).__init__()

	def forward(self, x):
		return x


class CNNDecoder(nn.Module):
	def __init__(self, hyperparams, hdim):
		super(CNNDecoder, self).__init__()

	def forward(self, x):
		return x
