import torch
import torch.nn as nn
import math


class CNNEncoder(nn.Module):
	def __init__(self, params, hdim, num_seq_labels):
		"""
		Based on paper: DeepTCR is a deep learning framework for revealing sequence concepts within T-cell repertoires
		:param params: hyperparameters as dict
		:param hdim: int, output feature dimension
		:param num_seq_labels: int, number of aa labels, input dim (feature dim)
		"""
		super(CNNEncoder, self).__init__()
		input_len = params['max_tcr_length']
		params = params['encoder']

		if type(params['kernel']) == int:
			kernel = [params['kernel']] * params['num_layers']
		elif type(params['kernel']) == list:
			kernel = params['kernel']
		else:
			raise ValueError()

		if type(params['num_features']) == int:
			num_features = [params['num_features']] * params['num_layers']
		elif type(params['num_features']) == list:
			num_features = params['num_features']
		else:
			raise ValueError()

		if type(params['stride']) == int:
			stride = [params['stride']] * params['num_layers']
		elif type(params['stride']) == list:
			stride = params['stride']
		else:
			raise ValueError()

		assert len(stride) == len(kernel)
		assert len(kernel) == len(num_features)

		self.embedding = nn.Embedding(num_embeddings=num_seq_labels, embedding_dim=params['embedding_dim'], padding_idx=0)

		input_features = [params['embedding_dim']] + num_features[:-1]
		output_len = input_len  # Sequence length after all convolution blocks
		cnn_blocks = []
		for input_, output_, kernel_, stride_ in zip(input_features, num_features, kernel, stride):
			padding = (kernel_ - 1) // 2
			output_len = math.floor((output_len + (2 * padding) - kernel_) / stride_ + 1)
			block = []
			block.append(nn.Conv1d(input_, output_, kernel_, stride_, padding=padding, bias=False if params['batch_norm'] else True))
			if params['batch_norm']:
				block.append(nn.BatchNorm1d(output_))
			if params['activation'] != 'linear':
				block.append(self._activation(params['activation']))
			if params['dropout'] is not None and params['dropout'] != 0.0:
				block.append(nn.Dropout(params['dropout']))
			cnn_blocks.append(nn.Sequential(*block))

		self.cnn_blocks = nn.Sequential(*cnn_blocks)
		self.output_layer = nn.Linear(num_features[-1] * output_len, hdim)

	def forward(self, tcr_seq, tcr_len):
		x = self.embedding(tcr_seq)  # shape=[batch, sequence, feature]
		x = x.permute(0, 2, 1)  # shape=[batch, feature, sequence]
		x = self.cnn_blocks(x)  # shape=[batch, feature, sequence]

		x = x.flatten(1)
		x = self.output_layer(x)

		return x

	def _activation(self, name):
		if name == 'relu':
			return nn.ReLU(inplace=True)
		elif name == 'leakyrelu':
			return nn.LeakyReLU(0.2, inplace=True)
		elif name == 'sigmoid':
			return nn.Sigmoid()
		elif name == 'softmax':
			return nn.Softmax()
		else:
			raise NotImplementedError(f'activation function {name} is not implemented.')


class CNNDecoder(nn.Module):
	def __init__(self, params, hdim, num_seq_labels):
		"""
		Based on paper: DeepTCR is a deep learning framework for revealing sequence concepts within T-cell repertoires
		:param params: hyperparameters as dict
		:param hdim: input feature dimension
		:param num_seq_labels: number of aa labels, output dim
		"""
		super(CNNDecoder, self).__init__()

		self.max_len = params['max_tcr_length']
		params = params['decoder']

		if type(params['kernel']) == int:
			kernel = [params['kernel']] * params['num_layers']
		elif type(params['kernel']) == list:
			kernel = params['kernel']
		else:
			raise ValueError()

		if type(params['num_features']) == int:
			num_features = [params['num_features']] * params['num_layers']
		elif type(params['num_features']) == list:
			num_features = params['num_features']
		else:
			raise ValueError()

		if type(params['stride']) == int:
			stride = [params['stride']] * params['num_layers']
		elif type(params['stride']) == list:
			stride = params['stride']
		else:
			raise ValueError()

		assert len(stride) == len(kernel)
		assert len(kernel) == len(num_features) - 1

		# out = (in - 1) * stride + kernel_size
		self.initial_len = params['initial_len']
		self.initial_feat_dim = params['num_features'][0]

		# The output of this layer will later be reshaped to [batch, num_feature of first deconv, initial_len]
		self.input_layer = nn.Linear(hdim, self.initial_feat_dim * self.initial_len)

		conv_transpose_blocks = []

		current_len = self.initial_len
		for input_, output_, kernel_, stride_ in zip(num_features[:-1], num_features[1:], kernel, stride):
			block = []
			current_len = (current_len - 1) * stride_ + kernel_
			block.append(nn.ConvTranspose1d(input_, output_, kernel_, stride_, bias=False if params['batch_norm'] else True))
			if params['batch_norm']:
				block.append(nn.BatchNorm1d(output_))
			if params['activation'] != 'linear':
				block.append(self._activation(params['activation']))
			if params['dropout'] is not None and params['dropout'] != 0.0:
				block.append(nn.Dropout(params['dropout']))
			conv_transpose_blocks.append(nn.Sequential(*block))

		# Last layer needs kernel and stride calculated beforehand
		kernel_, stride_ = self.get_kernel_stride(current_len, self.max_len)
		block = nn.ConvTranspose1d(num_features[-1], num_seq_labels, kernel_, stride_)
		conv_transpose_blocks.append(nn.Sequential(block))

		self.conv_transpose_blocks = nn.Sequential(*conv_transpose_blocks)

	def forward(self, x, gt_input):
		x = self.input_layer(x)
		x = x.reshape(-1, self.initial_feat_dim, self.initial_len)  # "unflatten" to create [batch, feature, seq_len]
		x = self.conv_transpose_blocks(x)

		x = x.permute(0, 2, 1)  # shape=[batch, seq_len, num_aa]
		x = x[:, :self.max_len]  # only take the max_len sequence, as the deconvolution can only create certain discreet lengths

		return x

	def get_kernel_stride(self, current_len, max_len):
		"""
		Determines the kernel and stride to have an output length that is higher than max_len
		https://github.com/sidhomj/DeepTCR/blob/db2c91a0422ef2c3281a594b3a7f1ac1009e3518/DeepTCR/functions/Layers.py#L240
		:params current_len:
		:params max_len:
		:return: (int, int)
		"""
		for kernel_ in list(range(4, 100)):
			for stride_ in list(range(2, 100)):
				len_after_conv_transpose = (current_len - 1) * stride_ + kernel_
				if len_after_conv_transpose >= max_len:
					return kernel_, stride_

		return AssertionError('Sequence is too long')

	def _activation(self, name):
		if name == 'relu':
			return nn.ReLU(inplace=True)
		elif name == 'leakyrelu':
			return nn.LeakyReLU(0.2, inplace=True)
		elif name == 'sigmoid':
			return nn.Sigmoid()
		elif name == 'softmax':
			return nn.Softmax()
		else:
			raise NotImplementedError(f'activation function {name} is not implemented.')


if __name__ == '__main__':
	import yaml
	with open('config/cnn_test.yaml') as file:
		params = yaml.load(file)
	params['seq_model_hyperparams']['max_tcr_length'] = 1000

	encoder = CNNEncoder(params['seq_model_hyperparams'], hdim=params['hdim'], num_seq_labels=25).cuda()
	input_ = torch.randint(25, (10, params['seq_model_hyperparams']['max_tcr_length'])).cuda()
	x = encoder(input_, 0)

	decoder = CNNDecoder(params['seq_model_hyperparams'], hdim=params['hdim'], num_seq_labels=25).cuda()
	output = decoder(x, 0)

	print(encoder)
	print(decoder)
	print(x.shape)
	print(output.shape)
