import torch
import os
import pandas as pd
import numpy as np
import random

class BaseModel:
	@property
	def history(self):
		return pd.DataFrame(self._val_history)

	@property
	def train_history(self):
		return pd.DataFrame(self._train_history)

	def save(self, filepath):
		""" Save model and optimizer state, and auxiliary data for continuing training """
		model_file = {'state_dict': self.model.state_dict(),
					  'train_history': self._train_history,
					  'val_history': self._val_history,
					  'epoch': self.epoch,
					  'aa_to_id': self.aa_to_id,
					  'params': self.params,
					  'best_loss': self.best_loss,
					  'train_masks': self.train_masks,
					  'best_cls_metric': self.best_cls_metric,
					  'best_knn_metric': self.best_knn_metric}
		try:
			model_file['optimizer'] = self.optimizer.state_dict()
		except:
			pass
		torch.save(model_file, filepath)

	def load(self, filepath):
		""" Load model and optimizer state, and auxiliary data for continuing training """
		model_file = torch.load(os.path.join(filepath))  # , map_location=self.device)
		# TODO Backwards compatibility, previous (before 25.03.2021) version didn't had theta in model, can be deleted later, change back strict=True
		missing_keys, unexpected_keys = self.model.load_state_dict(model_file['state_dict'], strict=False)
		if not (('theta' in missing_keys and len(missing_keys) == 1) or (
				len(missing_keys) == 0 and len(unexpected_keys) == 0)):
			if len(unexpected_keys) > 0:
				raise RuntimeError('Unexpected key(s) in state_dict: {}. '.format(
					', '.join('"{}"'.format(k) for k in unexpected_keys)))
			if len(missing_keys) > 0:
				raise RuntimeError(
					'Missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys)))

		self._train_history = model_file['train_history']
		self._val_history = model_file['val_history']
		self.epoch = model_file['epoch']
		self.aa_to_id = model_file['aa_to_id']
		self.best_loss = model_file['best_loss']
		self.train_masks = model_file['train_masks']
		if 'best_cls_metric' in model_file.keys():  # backward compatibility
			self.best_cls_metric = model_file['best_cls_metric']
		if 'best_knn_metric' in model_file.keys():  # backward compatibility
			self.best_knn_metric = model_file['best_knn_metric']

		try:
			self.optimizer.load_state_dict(model_file['optimizer'])
		except:
			pass

	# def save_data(self, filepath, train_masks):
	# 	data = {#'raw_data': [self.adatas, self.names, self.gene_layers, self.seq_keys],
	# 			'train_masks': train_masks}
	# 	torch.save(data, filepath)

	def load_data(self, filepath):
		""" Only load training masks """
		data_file = torch.load(os.path.join(filepath))  # , map_location=self.device)
		# self.adatas, self.names, self.gene_layers, self.seq_keys = data_file['raw_data']
		# it returns train masks
		return data_file['train_masks']

	def get_max_tcr_length(self):
		"""
		Determine the maximum amount of letters in the TCR sequence (TRA+TRB+codons)
		:return: int value maximal sequence length
		"""
		max_length = -99
		for adata in self.adatas:
			tcr_data = adata.obs['TRA+TRB']
			current_max = tcr_data.str.len().max()
			max_length = max(current_max, max_length)
		return max_length

	def kl_annealing(self, e, kl_annealing_epochs):
		"""
		Calculate KLD annealing factor, i.e. KLD needs to get warmup
		:param e: current epoch
		:param kl_annealing_epochs: total number of warmup epochs
		:return:
		"""
		return min(1.0, e / kl_annealing_epochs)

	def seed_worker(worker_id):
		worker_seed = torch.initial_seed() % 2 ** 32
		np.random.seed(worker_seed)
		random.seed(worker_seed)