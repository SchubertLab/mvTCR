import torch
import numpy as np
from scipy import sparse


class JointDataset(torch.utils.data.Dataset):
	def __init__(
			self,
			rna_data,
			tcr_data,
			tcr_length,
			metadata,
			labels=None,
			conditional=None
	):
		"""
		:param rna_data: list of gene expressions, where each element is a numpy or sparse matrix of one dataset
		:param tcr_data: list of seq_data, where each element is a seq_list of one dataset
		:param tcr_length: list of non-padded sequence length, needed for many architectures to mask the padding out
		:param metadata: list of metadata
		:param labels: list of labels
		:param conditional: list of conditionales
		"""
		self.metadata = metadata.tolist()
		self.tcr_length = torch.LongTensor(tcr_length)

		if conditional is not None:
			# Reduce the one-hot-encoding back to labels
			self.conditional = torch.LongTensor(conditional.argmax(1))
		# LongTensor since it is going to be embedded
		else:
			self.conditional = None

		self.rna_data = self.create_tensor(rna_data)
		# self.size_factors = self.rna_data.sum(1)

		self.tcr_data = torch.LongTensor(tcr_data)

		if labels is not None:
			self.labels = torch.LongTensor(self.labels)
		else:
			self.labels = None

	def create_tensor(self, x):
		if sparse.issparse(x):
			x = x.todense()
			return torch.FloatTensor(x)
		else:
			return torch.FloatTensor(x)

	def __len__(self):
		return len(self.rna_data)

	def __getitem__(self, idx):
		if self.labels is None:
			if self.conditional is None:
				return self.rna_data[idx], self.tcr_data[idx], self.tcr_length[idx], \
					   self.metadata[idx], False, False
			else:
				return self.rna_data[idx], self.tcr_data[idx], self.tcr_length[idx], \
					   self.metadata[idx], False, self.conditional[idx]
		else:
			if self.conditional is None:
				return self.rna_data[idx], self.tcr_datas[idx], self.tcr_lenght[idx], \
					   self.metadata[idx], self.labels[idx], False
			else:
				return self.rna_data[idx], self.tcr_datas[idx], self.tcr_lenght[idx], \
					   self.metadata[idx], self.labels[idx], self.conditional[idx]


class DeepTCRDataset(torch.utils.data.Dataset):
	def __init__(
			self,
			alpha_seq,
			beta_seq,
			vdj_dict,
			metadata
	):
		"""
		:param alpha_seq:
		:param beta_seq:
		:param vdj_dict:
		:param metadata: list of metadata
		"""
		self.metadata = np.concatenate(metadata, axis=0).tolist()

		# Concatenate datasets to be able to shuffle data through

		self.alpha_seq = np.concatenate(alpha_seq)
		self.alpha_seq = torch.LongTensor(self.alpha_seq)

		self.beta_seq = np.concatenate(beta_seq)
		self.beta_seq = torch.LongTensor(self.beta_seq)

		v_alpha = torch.LongTensor(vdj_dict['v_alpha'])
		j_alpha = torch.LongTensor(vdj_dict['j_alpha'])

		v_beta = torch.LongTensor(vdj_dict['v_beta'])
		d_beta = torch.LongTensor(vdj_dict['d_beta'])
		j_beta = torch.LongTensor(vdj_dict['j_beta'])

		self.vdj = torch.stack([v_alpha, j_alpha, v_beta, d_beta, j_beta], dim=1)

	def __len__(self):
		return len(self.alpha_seq)

	def __getitem__(self, idx):
		return self.alpha_seq[idx], self.beta_seq[idx], self.vdj[idx], self.metadata[idx]
