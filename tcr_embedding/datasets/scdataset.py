import torch
import numpy as np
from scipy import sparse

class TCRDataset(torch.utils.data.Dataset):
	def __init__(
			self,
			scRNA_datas,
			seq_datas,
			seq_len,
			adatas,
			dataset_names,
			index_list,
			metadata,
			labels=None
	):
		"""
		:param scRNA_datas: list of gene expressions, where each element is a numpy or sparse matrix of one dataset
		:param seq_datas: list of seq_data, where each element is a seq_list of one dataset
		:param seq_len: list of non-padded sequence length, needed for many architectures to mask the padding out
		:param adatas: list of raw adata
		:param dataset_names: list of concatenated dataset names with len=sum of dataset-lengths
		:param metadata: list of metadata
		:param labels: list of labels
		"""
		self.adatas = adatas
		self.dataset_names = dataset_names
		self.index_list = index_list
		self.seq_len = seq_len
		self.metadata = np.concatenate(metadata, axis=0).tolist()

		# Concatenate datasets to be able to shuffle data through
		self.scRNA_datas = []
		for scRNA_data in scRNA_datas:
			self.scRNA_datas.append(self._create_tensor(scRNA_data))
		self.scRNA_datas = torch.cat(self.scRNA_datas, dim=0)
		self.size_factors = self.scRNA_datas.sum(1)

		self.seq_datas = np.concatenate(seq_datas)
		self.seq_datas = torch.LongTensor(self.seq_datas)

		if labels is not None:
			self.labels = np.concatenate(labels)
			self.labels = torch.LongTensor(self.labels)
		else:
			self.labels = None

	def _create_tensor(self, x):
		if sparse.issparse(x):
			x = x.todense()
			return torch.FloatTensor(x)
		else:
			return torch.FloatTensor(x)

	def __len__(self):
		return len(self.scRNA_datas)

	def __getitem__(self, idx):
		# arbitrary additional info from adatas
		# self.adatas[self.dataset_names[idx]].obs.loc[self.index_list[idx]][column]
		if self.labels is None:
			return self.scRNA_datas[idx], self.seq_datas[idx], self.size_factors[idx], self.dataset_names[idx], self.index_list[idx], self.seq_len[idx], self.metadata[idx]
		else:
			return self.scRNA_datas[idx], self.seq_datas[idx], self.size_factors[idx], self.dataset_names[idx], self.index_list[idx], self.seq_len[idx], self.metadata[idx], self.labels[idx]


class DeepTCRDataset(torch.utils.data.Dataset):
	def __init__(
			self,
			alpha_seq,
			beta_seq,
			vdj_dict,
			adatas,
			dataset_names,
			index_list,
			metadata
	):
		"""
		:param scRNA_datas: list of gene expressions, where each element is a numpy or sparse matrix of one dataset
		:param seq_datas: list of seq_data, where each element is a seq_list of one dataset
		:param seq_len: list of non-padded sequence length, needed for many architectures to mask the padding out
		:param adatas: list of raw adata
		:param dataset_names: list of concatenated dataset names with len=sum of dataset-lengths
		:param metadat: list of metadata
		"""
		self.adatas = adatas
		self.dataset_names = dataset_names
		self.index_list = index_list
		self.metadata = np.concatenate(metadata, axis=0).tolist()

		# Concatenate datasets to be able to shuffle data through

		self.alpha_seq = np.concatenate(alpha_seq)
		self.alpha_seq = torch.LongTensor(self.alpha_seq)

		self.beta_seq = np.concatenate(beta_seq)
		self.beta_seq = torch.LongTensor(self.beta_seq)

		v_alpha = np.concatenate(vdj_dict['v_alpha'])
		v_alpha = torch.LongTensor(v_alpha)

		j_alpha = np.concatenate(vdj_dict['j_alpha'])
		j_alpha = torch.LongTensor(j_alpha)

		v_beta = np.concatenate(vdj_dict['v_beta'])
		v_beta = torch.LongTensor(v_beta)

		d_beta = np.concatenate(vdj_dict['d_beta'])
		d_beta = torch.LongTensor(d_beta)

		j_beta = np.concatenate(vdj_dict['j_beta'])
		j_beta = torch.LongTensor(j_beta)

		self.vdj = torch.stack([v_alpha, j_alpha, v_beta, d_beta, j_beta], dim=1)

	def __len__(self):
		return len(self.alpha_seq)

	def __getitem__(self, idx):
		# arbitrary additional info from adatas
		# self.adatas[self.dataset_names[idx]].obs.loc[self.index_list[idx]][column]
		return self.alpha_seq[idx], self.beta_seq[idx], self.vdj[idx], \
			   self.dataset_names[idx], self.index_list[idx], self.metadata[idx]
