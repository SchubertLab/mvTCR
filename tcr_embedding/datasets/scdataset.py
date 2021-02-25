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
		self.seq_len = seq_len
		self.metadata = np.concatenate(metadata, axis=0).tolist()

		# Concatenate datasets to be able to shuffle data through
		self.scRNA_datas = []
		for scRNA_data in scRNA_datas:
			self.scRNA_datas.append(self._create_tensor(scRNA_data))
		self.scRNA_datas = torch.cat(self.scRNA_datas, dim=0)

		self.seq_datas = [item for sublist in seq_datas for item in sublist.to_list()]

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
		return self.scRNA_datas[idx], self.seq_datas[idx], self.dataset_names[idx], self.index_list[idx], self.seq_len[idx], self.metadata[idx]
