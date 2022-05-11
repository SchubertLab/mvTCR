import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import scanpy as sc
from abc import ABC, abstractmethod

from .losses.kld import KLD

from tcr_embedding.dataloader.DataLoader import initialize_data_loader, initialize_latent_loader
from tcr_embedding.dataloader.DataLoader import initialize_prediction_loader

from .optimization.knn_prediction import report_knn_prediction
from .optimization.modulation_prediction import report_modulation_prediction
from .optimization.pseudo_metric import report_pseudo_metric


class VAEBaseModel(ABC):
	def __init__(self,
				 adata,
				 params_architecture,
				 balanced_sampling='clonotype',
				 metadata=None,
				 conditional=None,
				 optimization_mode_params=None,
				 label_key=None,
				 device=None):
		"""
		VAE Base Model, used for both single and joint models
		:param adata: list of adatas containing train and val set
		:param conditional: str or None, if None a normal VAE is used, if str then the str determines the adata.obsm[conditional] as conditioning variable
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param balanced_sampling: None or str, indicate adata.obs column to balance
		:param optimization_mode_params: dict carrying the mode specific parameters
		"""
		self.adata = adata
		self.params_architecture = params_architecture
		self.balanced_sampling = balanced_sampling
		self.metadata = metadata
		self.conditional = conditional
		self.optimization_mode_params = optimization_mode_params
		self.label_key = label_key
		self.device = device

		self.params_tcr = None
		self.params_rna = None
		self.params_joint = params_architecture['joint']
		self.params_supervised = None
		self.beta_only = False

		if 'tcr' in params_architecture:
			self.params_tcr = params_architecture['tcr']
			if 'beta_only' in self.params_tcr:
				self.beta_only = self.params_tcr['beta_only']
		if 'rna' in params_architecture:
			self.params_rna = params_architecture['rna']
		if 'supervised' in params_architecture:
			self.params_supervised = params_architecture['supervised']

		if self.params_tcr is None and self.params_rna is None:
			raise ValueError('Please specify either tcr, rna, or both hyperparameters.')

		self.aa_to_id = adata.uns['aa_to_id']

		if self.device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self._train_history = defaultdict(list)
		self._val_history = defaultdict(list)

		# counters
		self.best_optimization_metric = None
		self.best_loss = None
		self.no_improvements = 0

		# loss functions
		self.loss_function_rna = nn.MSELoss()
		self.loss_function_tcr = nn.CrossEntropyLoss(ignore_index=self.aa_to_id['_'])
		self.loss_function_kld = KLD()
		self.loss_function_class = nn.CrossEntropyLoss()

		# training params
		self.batch_size = params_architecture['batch_size']
		self.loss_weights = None
		self.comet = None
		self.kl_annealing_epochs = None

		# Model
		self.model_type = None
		self.model = None
		self.optimizer = None
		self.supervised_model = None
		if self.label_key is not None:
			assert self.params_supervised is not None, 'Please specify parameters for supervised model'
			self.supervised_model = None  # todo

		# datasets
		if metadata is None:
			metadata = []
		if balanced_sampling is not None and balanced_sampling not in metadata:
			metadata.append(balanced_sampling)
		self.data_train, self.data_val = initialize_data_loader(adata, metadata, conditional, label_key,
																balanced_sampling, self.batch_size,
																beta_only=self.beta_only)

	def change_adata(self, new_adata):
		self.adata = new_adata
		self.aa_to_id = new_adata.uns['aa_to_id']
		if self.balanced_sampling is not None and self.balanced_sampling not in self.metadata:
			self.metadata.append(self.balanced_sampling)

		self.data_train, self.data_val = initialize_data_loader(new_adata, self.metadata, self.conditional, self.label_key,
																self.balanced_sampling, self.batch_size,
																beta_only=self.beta_only)

	def add_new_embeddings(self, num_new_embeddings):
		cond_emb_tmp = self.model.cond_emb.weight.data
		self.model.cond_emb = torch.nn.Embedding(cond_emb_tmp.shape[0]+num_new_embeddings, cond_emb_tmp.shape[1])
		self.model.cond_emb.weight.data[:cond_emb_tmp.shape[0]] = cond_emb_tmp

	def freeze_all_weights_except_cond_embeddings(self):
		"""
		Freezes conditional embedding weights to train in scArches style, since training data doesn't include
		previous labels, those embeddings won't be updated
		"""
		for param in self.model.parameters():
			param.requires_grad = False

		self.model.cond_emb.weight.requires_grad = True

	def unfreeze_all(self):
		for param in self.model.parameters():
			param.requires_grad = True

	def train(self,
			  n_epochs=200,
			  batch_size=512,
			  learning_rate=3e-4,
			  loss_weights=None,
			  kl_annealing_epochs=None,
			  early_stop=None,
			  save_path='../saved_models/',
			  comet=None):
		"""
		Train the model for n_epochs
		:param n_epochs: None or int, number of epochs to train, if None n_iters needs to be int
		:param batch_size: int, batch_size
		:param learning_rate: float, learning rate
		:param loss_weights: list of floats, loss_weights[0]:=weight or scRNA loss, loss_weights[1]:=weight for TCR loss, loss_weights[2] := KLD Loss
		:param kl_annealing_epochs: int or None, int number of epochs until kl reaches maximum warmup, if None this value is set to 30% of n_epochs
		:param early_stop: int, stop training after this number of epochs if val loss is not improving anymore
		:param save_path: str, path to directory to save model
		:param comet: None or comet_ml.Experiment object
		:return:
		"""
		self.batch_size = batch_size
		self.loss_weights = loss_weights
		self.comet = comet
		self.kl_annealing_epochs = kl_annealing_epochs
		assert 3 <= len(loss_weights) <= 4, 'Length of loss weights need to be either 3 or 4.'

		try:
			os.makedirs(save_path)  # Create directory to prevent Error while saving model weights
		except OSError:
			pass

		if kl_annealing_epochs is None:
			self.kl_annealing_epochs = int(0.3 * n_epochs)

		# raise ValueError(f'length of loss_weights must be 3, 4 (supervised) or None.')

		self.model = self.model.to(self.device)
		self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)

		for epoch in tqdm(range(n_epochs)):
			self.model.train()
			train_loss_summary = self.run_epoch(epoch, phase='train')
			self.log_losses(train_loss_summary, epoch)

			self.model.eval()
			with torch.no_grad():
				val_loss_summary = self.run_epoch(epoch, phase='val')
				self.log_losses(val_loss_summary, epoch)
				self.additional_evaluation(epoch, save_path)

			if self.do_early_stopping(val_loss_summary['val Loss'], early_stop, save_path, epoch):
				break

	def run_epoch(self, epoch, phase='train'):
		if phase == 'train':
			data = self.data_train
		else:
			data = self.data_val
		loss_total = []
		rna_loss_total = []
		tcr_loss_total = []
		kld_loss_total = []
		cls_loss_total = []
		cls_acc_total = []

		for rna, tcr, seq_len, _, labels, conditional in data:
			if rna.shape[0] == 1 and phase == 'train':
				continue  # BatchNorm cannot handle batches of size 1 during training phase
			rna = rna.to(self.device)
			tcr = tcr.to(self.device)

			if self.conditional is not None:
				conditional = conditional.to(self.device)
			else:
				conditional = None

			z, mu, logvar, rna_pred, tcr_pred = self.model(rna, tcr, seq_len, conditional)
			kld_loss, z = self.calculate_kld_loss(mu, logvar, epoch)
			rna_loss, tcr_loss = self.calculate_loss(rna_pred, rna, tcr_pred, tcr)
			loss = kld_loss + rna_loss + tcr_loss

			if self.supervised_model is not None:
				prediction_label = self.forward_supervised(z)
				cls_acc = torch.sum(torch.eq(prediction_label, labels))
				cls_loss = self.calculate_classification_loss(prediction_label, labels)
				loss += cls_loss
				cls_loss_total.append(cls_loss)
				cls_acc_total.append(cls_acc)

			if phase == 'train':
				self.run_backward_pass(loss)

			loss_total.append(loss)
			rna_loss_total.append(rna_loss)
			tcr_loss_total.append(tcr_loss)
			kld_loss_total.append(kld_loss)

			if torch.isnan(loss):
				print(f'ERROR: NaN in loss.')
				return

		loss_total = torch.stack(loss_total).mean().item()
		rna_loss_total = torch.stack(rna_loss_total).mean().item()
		tcr_loss_total = torch.stack(tcr_loss_total).mean().item()
		kld_loss_total = torch.stack(kld_loss_total).mean().item()

		summary_losses = {f'{phase} Loss': loss_total,
						  f'{phase} RNA Loss': rna_loss_total,
						  f'{phase} TCR Loss': tcr_loss_total,
						  f'{phase} KLD Loss': kld_loss_total}

		if self.supervised_model is not None:
			cls_loss_total = torch.stack(cls_loss_total).mean().item()
			summary_losses[f'{phase} CLS Loss'] = cls_loss_total
			summary_losses[f'{phase} CLS Accuracy'] = cls_acc_total

		return summary_losses

	def log_losses(self, summary_losses, epoch):
		if self.comet is not None:
			self.comet.log_metrics(summary_losses, epoch=epoch)

	def run_backward_pass(self, loss):
		self.optimizer.zero_grad()
		loss.backward()
		if self.optimization_mode_params is not None and 'grad_clip' in self.optimization_mode_params:
			nn.utils.clip_grad_value_(self.model.parameters(), self.optimization_mode_params['grad_clip'])
		self.optimizer.step()

	def additional_evaluation(self, epoch, save_path):
		if self.optimization_mode_params is None:
			return
		name = self.optimization_mode_params['name']
		if name == 'reconstruction':
			return
		if name == 'knn_prediction':
			score, relation = report_knn_prediction(self.adata, self, self.optimization_mode_params,
													epoch, self.comet)
		elif name == 'modulation_prediction':
			score, relation = report_modulation_prediction(self.adata, self, self.optimization_mode_params,
														   epoch, self.comet)
		elif name == 'pseudo_metric':
			score, relation = report_pseudo_metric(self.adata, self, self.optimization_mode_params,
												   epoch, self.comet)
		else:
			raise ValueError('Unknown Optimization mode')
		if self.best_optimization_metric is None or relation(score, self.best_optimization_metric):
			self.best_optimization_metric = score
			self.save(os.path.join(save_path, f'best_model_by_metric.pt'))
		if self.comet is not None:
			self.comet.log_metric('max_metric', self.best_optimization_metric, epoch=epoch)

	def do_early_stopping(self, val_loss, early_stop, save_path, epoch):
		if self.best_loss is None or val_loss < self.best_loss:
			self.best_loss = val_loss
			self.save(os.path.join(save_path, 'best_model_by_reconstruction.pt'))
			self.no_improvements = 0
		else:
			self.no_improvements += 1
		if early_stop is not None and self.no_improvements > early_stop:
			print('Early stopped')
			return True
		if self.comet is not None:
			self.comet.log_metric('Epochs without Improvements', self.no_improvements, epoch=epoch)
		return False

	# <- prediction functions ->
	def get_latent(self, adata, metadata, return_mean=True):
		"""
		Get latent
		:param adata:
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param return_mean: bool, calculate latent space without sampling
		:return: adata containing embedding vector in adata.X for each cell and the specified metadata in adata.obs
		"""
		data_embed = initialize_prediction_loader(adata, metadata, self.batch_size, beta_only=self.beta_only,
												  conditional=self.conditional)

		zs = []
		with torch.no_grad():
			self.model = self.model.to(self.device)
			self.model.eval()
			for rna, tcr, seq_len, _, labels, conditional in data_embed:
				rna = rna.to(self.device)
				tcr = tcr.to(self.device)

				if self.conditional is not None:
					conditional = conditional.to(self.device)
				else:
					conditional = None
				z, mu, _, _, _ = self.model(rna, tcr, seq_len, conditional)
				if return_mean:
					z = mu
				z = self.model.get_latent_from_z(z)
				z = sc.AnnData(z.detach().cpu().numpy())
				# z.obs[metadata] = np.array(metadata_batch).T
				zs.append(z)
		latent = sc.AnnData.concatenate(*zs)
		latent.obs.index = adata.obs.index
		latent.obs[metadata] = adata.obs[metadata]
		return latent

	def predict_rna_from_latent(self, adata_latent, metadata=None):
		data = initialize_latent_loader(adata_latent, self.batch_size, self.conditional)
		rnas = []
		with torch.no_grad():
			model = self.model.to(self.device)
			model.eval()
			for batch in data:
				if self.conditional is not None:
					batch = batch[0].to(self.device)
					conditional = batch[1].to(self.device)
				else:
					batch = batch[0].to(self.device)
					conditional = None
				batch_rna = model.predict_transcriptome(batch, conditional)
				batch_rna = sc.AnnData(batch_rna.detach().cpu().numpy())
				rnas.append(batch_rna)
		rnas = sc.AnnData.concatenate(*rnas)
		rnas.obs.index = adata_latent.obs.index
		if metadata is not None:
			rnas.obs[metadata] = adata_latent.obs[metadata]
		return rnas

	def predict_label(self, adata):
		data, _ = initialize_data_loader(adata, None, self.conditional, self.label_key,
										 None, self.batch_size, beta_only=self.beta_only)
		prediction_total = []
		with torch.no_grad():
			for rna, tcr, seq_len, metadata_batch, labels, conditional in data:
				rna = rna.to(self.device)
				tcr = tcr.to(self.device)

				if self.conditional is not None:
					conditional = conditional.to(self.device)
				else:
					conditional = None

				z = self.model(rna, tcr, seq_len, conditional)
				prediction = self.forward_supervised(z)
				prediction_total.append(prediction)
		prediction_total = torch.stack(prediction_total).cpu().detach().numpy()
		return prediction_total

	# <- semi-supervised model ->
	def forward_supervised(self, z):
		z_ = self.model.get_latent_from_z(z)
		prediction = self.supervised_model(z_)
		raise prediction

	# <- loss functions ->
	@abstractmethod
	def calculate_loss(self, rna_pred, rna, tcr_pred, tcr):
		raise NotImplementedError

	@abstractmethod
	def calculate_kld_loss(self, mu, logvar, epoch):
		"""
		Calculate the kld loss and z depending on the model type
		:param mu: mean of the VAE latent space
		:param logvar: log(variance) of the VAE latent space
		:param epoch: current epoch as integer
		:return:
		"""
		raise NotImplementedError('Implement this in the different model versions')

	def calculate_classification_loss(self, prediction, labels):
		loss = self.loss_function_class(prediction, labels)
		loss = self.loss_weights[3] * loss
		return loss

	def get_kl_annealing_factor(self, epoch):
		"""
		Calculate KLD annealing factor, i.e. KLD needs to get warmup
		:param epoch: current epoch
		:return:
		"""
		return min(1.0, epoch / self.kl_annealing_epochs)

	# <- logging helpers ->
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
					  'aa_to_id': self.aa_to_id,

					  'params_architecture': self.params_architecture,
					  'balanced_sampling': self.balanced_sampling,
					  'metadata': self.metadata,
					  'conditional': self.conditional,
					  'optimization_mode_params': self.optimization_mode_params,
					  'label_key': self.label_key,
					  'model_type': self.model_type,
					  }
		torch.save(model_file, filepath)

	def load(self, filepath, map_location=torch.device('cuda')):
		""" Load model for evaluation / inference"""
		model_file = torch.load(os.path.join(filepath), map_location=map_location)
		self.model.load_state_dict(model_file['state_dict'], strict=False)
		self._train_history = model_file['train_history']
		self._val_history = model_file['val_history']
		self.aa_to_id = model_file['aa_to_id']
