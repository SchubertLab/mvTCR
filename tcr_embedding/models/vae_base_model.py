import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import scanpy as sc
from sklearn.metrics import classification_report
from abc import ABC, abstractmethod

from tcr_embedding.datasets.scdataset import TCRDataset
from .losses.nb import NB
from .losses.kld import KLD

from tcr_embedding.models.mixture_modules.base_model import BaseModel
from tcr_embedding.evaluation.kNN import run_knn_within_set_evaluation
from tcr_embedding.evaluation.Imputation import run_imputation_evaluation
from tcr_embedding.evaluation.WrapperFunctions import get_model_prediction_function
from tcr_embedding.models.scGen import run_scgen_cross_validation


class VAEBaseModel(BaseModel, ABC):
	def __init__(self,
				 adata,
				 aa_to_id,
				 params_architecture,
				 model_type='poe',
				 conditional=None,
				 optimization_mode='Reconstruction',
				 optimization_mode_params=None,
				 device=None,
				 ):
		"""
		VAE Base Model, used for both single and joint models
		:param adata: list of adatas containing train and val set
		:param aa_to_id: dict containing mapping from amino acid symbol to label idx
		:param conditional: str or None, if None a normal VAE is used, if str then the str determines the adata.obsm[conditional] as conditioning variable
		:param optimization_mode: str, 'Reconstruction', 'Prediction', 'scGen', 'RNA_KLD', 'RNA_MMD', 'RNA_MEAN'
		:param optimization_mode_params: dict carrying the mode specific parameters
		"""

		self.adata = adata

		self.params_tcr = None
		self.params_rna = None
		self.params_joint = params_architecture['joint']

		if 'tcr' in params_architecture:
			self.params_tcr = params_architecture['tcr']
		if 'rna' in params_architecture:
			self.params_rna = params_architecture['rna']

		if self.params_tcr is None and self.params_rna is None:
			raise ValueError('Please specify either tcr, rna, or both hyperparameters.')

		self.model_type = model_type.lower()
		self.aa_to_id = aa_to_id

		self.conditional = conditional
		self.optimization_mode = optimization_mode
		self.optimization_mode_params = optimization_mode_params

		self.device = device
		if self.device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self._train_history = defaultdict(list)
		self._val_history = defaultdict(list)

		# self.max_seq_length = self.get_max_tcr_length()
		self.kl_annealing_epochs = None
		self.best_optimization_metric = None
		self.best_loss = None
		self.no_improvements = 0

		# loss functions
		self.loss_function_rna = nn.MSELoss()
		self.loss_function_tcr = nn.CrossEntropyLoss(ignore_index=self.aa_to_id['_'])
		self.loss_function_kld = KLD()
		self.loss_function_class = nn.CrossEntropyLoss()


	def train(self,
			  n_epochs=100,
			  batch_size=64,
			  lr=3e-4,
			  loss_weights=None,
			  kl_annealing_epochs=None,
			  val_split='set',
			  metadata=None,
			  early_stop=None,
			  balanced_sampling=None,
			  save_path='../saved_models/',
			  comet=None,
			  ):
		"""
		Train the model for n_epochs
		:param n_epochs: None or int, number of epochs to train, if None n_iters needs to be int
		:param batch_size: int, batch_size
		:param lr: float, learning rate
		:param loss_weights: list of floats, loss_weights[0]:=weight or scRNA loss, loss_weights[1]:=weight for TCR loss, loss_weights[2] := KLD Loss
		:param kl_annealing_epochs: int or None, int number of epochs until kl reaches maximum warmup, if None this value is set to 30% of n_epochs
		:param val_split: str or float, if str it indicates the adata.obs[val_split] containing 'train' and 'val', if float, adata is split randomly by val_split
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param early_stop: int, stop training after this number of epochs if val loss is not improving anymore
		:param balanced_sampling: None or str, indicate adata.obs column to balance
		:param save_path: str, path to directory to save model
		:param comet: None or comet_ml.Experiment object
		:return:
		"""
		if self.device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		if metadata is None:
			metadata = []

		if balanced_sampling is not None and balanced_sampling not in metadata:
			metadata.append(balanced_sampling)

		try:
			os.makedirs(save_path)  # Create directory to prevent Error while saving model weights
		except:
			pass

		if kl_annealing_epochs is None:
			kl_annealing_epochs = int(0.3 * n_epochs)


		# raise ValueError(f'length of loss_weights must be 3, 4 (supervised) or None.')

		self.model = self.model.to(self.device)
		self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)


		for epoch in tqdm(n_epochs):
			self.model.train()
			train_loss_summary = self.run_epoch(epoch, phase='train')
			self.log_losses(train_loss_summary)
			self.run_backward_pass(total_loss_train)

			self.model.eval()
			with torch.no_grad():
				val_loss_summary = self.run_epoch(phase='val')
			self.log_losses(val_loss_summary)
			if self.do_early_stopping(total_loss_val):
				break


	def run_epoch(self, epoch, phase='train'):
		if phase == 'train':
			data = train_dataloader
		else:
			data = val_dataloader

		loss_total = []
		rna_loss_total = []
		tcr_loss_total = []
		kld_loss_total = []

		for scRNA, tcr_seq, size_factor, index, seq_len, metadata_batch, labels, conditional in data:
			scRNA = scRNA.to(self.device)
			tcr_seq = tcr_seq.to(self.device)

			if self.conditional is not None:
				conditional = conditional.to(self.device)
			else:
				conditional = None

			z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, tcr_seq, seq_len, conditional)
			kld_loss, z = self.calculate_kld_loss(loss_weights, mu, logvar, epoch)
			loss, rna_loss, tcr_loss = self.calculate_loss(scRNA_pred, scRNA, tcr_seq_pred, tcr_seq,
															 loss_weights, size_factor)

			loss = loss + kld_loss # todo rna_loss + tcr_loss

			loss_total.append(loss)
			rna_loss_total.append(rna_loss)
			tcr_loss_total.append(tcr_loss)
			kld_loss_total.append(kld_loss)

		loss_total = torch.stack(loss_total).mean().item()
		rna_loss_total = torch.stack(rna_loss_total).mean().item()
		tcr_loss_total = torch.stack(tcr_loss_total).mean().item()
		kld_loss_total = torch.stack(kld_loss_total).mean().item()

		if torch.isnan(loss_total):
			print(f'Loss became NaN, Loss: {loss}')
			return

		summary_losses = {f'{phase} Loss': loss_total,
						  f'{phase} scRNA Loss': rna_loss_total,
						  f'{phase} TCR Loss': tcr_loss_total,
						  f'{phase} KLD Loss': kld_loss_total}
		return summary_losses

	def log_losses(self, summary_losses, epoch):
		if comet is not None:
			comet.log_metrics(summary_losses,
							  epoch=epoch)

	def run_backward_pass(self, loss):
		self.optimizer.zero_grad()
		loss.backward()
		if self.optimization_mode_params is not None and 'grad_clip' in self.optimization_mode_params:
			nn.utils.clip_grad_value_(self.model.parameters(), self.optimization_mode_params['grad_clip'])
		self.optimizer.step()

	def additional_evaluation(self):
		if self.optimization_mode == 'Prediction':
			self.report_validation_prediction(batch_size, epoch, comet, save_path)
		if self.optimization_mode == 'Reconstruction':
			self.report_reconstruction(loss_val_total)
		if self.optimization_mode == 'PseudoMetric':
			self.report_pseudo_metric(batch_size, epoch, comet, save_path)
		if self.optimization_mode == 'scGen':
			self.report_scgen(batch_size, epoch, comet, save_path)

	def do_early_stopping(self):
		if (loss_val_total < self.best_loss):
			self.best_loss = loss_val_total
			self.save(os.path.join(save_path, 'best_reconstruction_model.pt'))
			no_improvements = 0
		else:
			no_improvements += 1
		if early_stop is not None and no_improvements > early_stop:
			print('Early stopped')
			break
		# logging
		'Epochs without Improvements': no_improvements},
		epoch = epoch


	def report_validation_prediction(self, batch_size, epoch, comet, save_path):
		"""
		Report the objective metric of the 10x dataset for hyper parameter optimization.
		:param batch_size: Batch size for creating the validation latent space
		:param epoch: epoch number for logging
		:param comet: Comet experiments for logging validation
		:param save_path: Path for saving trained models
		:return: Reports externally to comet, saves model.
		"""
		test_embedding_func = get_model_prediction_function(self, batch_size=batch_size)
		try:
			summary = run_imputation_evaluation(self.adata, test_embedding_func, query_source='val',
												use_non_binder=True, use_reduced_binders=True,
												label_pred=self.optimization_mode_params['prediction_column'])
		except:
			print(f'kNN did not work')
			return

		metrics = summary['knn']

		if comet is not None:
			for antigen, metric in metrics.items():
				if antigen != 'accuracy':
					comet.log_metrics(metric, prefix=antigen, epoch=epoch)
				else:
					comet.log_metric('accuracy', metric, epoch=epoch)

		if metrics['weighted avg']['f1-score'] > self.best_optimization_metric:
			self.best_optimization_metric = metrics['weighted avg']['f1-score']
			self.save(os.path.join(save_path, f'best_knn_model.pt'))

	def report_reconstruction(self, validation_loss):
		"""
		Report the reconstruction loss as metric for hyper parameter optimization.
		:param validation_loss: Reconstruction loss on the validation set
		:return: Reports to files
		"""
		# todo
		pass

	def report_pseudo_metric(self, batch_size, epoch, comet, save_path):
		"""
		Calculate a pseudo metric based on kNN of multiple meta information
		:param batch_size:
		:param epoch:
		:param comet:
		:param save_path:
		:return:
		"""
		test_embedding_func = get_model_prediction_function(self, batch_size=batch_size, do_adata=True,
														metadata=self.optimization_mode_params['prediction_labels'])
		try:
			summary = run_knn_within_set_evaluation(self.adata, test_embedding_func,
													self.optimization_mode_params['prediction_labels'], subset='val')
			summary['pseudo_metric'] = sum(summary.values())
		except Exception as e:
			print(e)
			print('Error in kNN')
			return

		if comet is not None:
			comet.log_metrics(summary, epoch=epoch)
		if self.best_optimization_metric is None or summary['pseudo_metric'] > self.best_optimization_metric:
			self.best_optimization_metric = summary['pseudo_metric']
			self.save(os.path.join(save_path, f'best_model_by_metric.pt'))
			comet.log_metric('max_pseudo_metric',  self.best_optimization_metric,
							 step=, epoch=epoch)

	def report_scgen(self, batch_size, epoch, comet, save_path):
		summary = run_scgen_cross_validation(self.adata, self.optimization_mode_params['column_fold'],
											 self, self.optimization_mode_params['column_perturbation'],
											 self.optimization_mode_params['indicator_perturbation'],
											 self.optimization_mode_params['column_cluster'])
		score = summary['avg_r_squared']

		if comet is not None:
			for key, value in summary.items():

				if key == 'avg_r_squared':
					comet.log_metric(key, value, step=int(epoch), epoch=epoch)
				else:
					comet.log_metrics(value, prefix=key, step=int(epoch), epoch=epoch)
		if self.best_optimization_metric is None or score > self.best_optimization_metric:
			self.best_optimization_metric = score
			self.save(os.path.join(save_path, 'best_model_by_scGen.pt'))
			if comet is not None:
				comet.log_metric('max_scGen_avg_R_squared',  self.best_optimization_metric,
								 step=int(epoch), epoch=epoch)

	def get_latent(self, adata, batch_size=256, num_workers=0, metadata=[], return_mean=True):
		"""
		Get latent
		:param adata:
		:param batch_size: int, batch size
		:param num_workers: int, num_workers for dataloader
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param return_mean: bool, calculate latent space without sampling
		:return: adata containing embedding vector in adata.X for each cell and the specified metadata in adata.obs
		"""
		pred_datasets, _, _ = self.create_datasets(adata, val_split=0, metadata=metadata)
		pred_dataloader = DataLoader(pred_datasets, batch_size=batch_size, shuffle=False, collate_fn=None,
									 num_workers=num_workers)

		zs = []
		with torch.no_grad():
			self.model = self.model.to(self.device)
			self.model.eval()
			for scRNA, tcr_seq, size_factor, name, index, seq_len, metadata_batch, labels, conditional in pred_dataloader:
				scRNA = scRNA.to(self.device)
				tcr_seq = tcr_seq.to(device)
				if self.conditional is not None:
					conditional = conditional.to(self.device)
				else:
					conditional = None
				z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, tcr_seq, seq_len, conditional)
				if return_mean:
					z = mu

				if self.moe:
					z = 0.5 * (mu[0] + mu[1])  # mean of latent space from both modalities for mmvae
				elif self.poe:
					z = z[2]  # use joint latent variable
				z = sc.AnnData(z.detach().cpu().numpy())
				z.obs['barcode'] = index
				z.obs['dataset'] = name
				z.obs[metadata] = np.array(metadata_batch).T
				zs.append(z)

		return sc.AnnData.concatenate(*zs)

	def predict_rna_from_latent(self, adata_latent, metadata=None):
		if self.conditional is None:
			dataset = torch.utils.data.TensorDataset(torch.from_numpy(adata_latent.X))
		else:
			dataset = torch.utils.data.TensorDataset(torch.from_numpy(adata_latent.X),
													 torch.from_numpy(adata_latent.obsm[self.conditional]))
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024)
		rnas = []
		with torch.no_grad():
			model = self.model.to(self.device)
			model.eval()
			for batch in dataloader:
				if self.conditional is not None:
					batch = batch[0].to(self.device)
					conditional = batch[1].to(self.device)
				else:
					batch = batch[0].to(self.device)
					conditional = None
				batch_rna = model.predict_rna(batch, conditional)
				batch_rna = sc.AnnData(batch_rna.detach().cpu().numpy())
				rnas.append(batch_rna)
		rnas = sc.AnnData.concatenate(*rnas)
		if metadata is not None:
			rnas.obs[metadata] = adata_latent.obs[metadata]
		return rnas

	def calculate_sampling_weights(self, adata, train_mask, class_column):
		"""
		Calculate sampling weights for more balanced sampling in case of imbalanced classes,
		:params class_column: str, key for class to be balanced
		:params log_divisor: divide the label counts by this factor before taking the log, higher number makes the sampling more uniformly balanced
		:return: list of weights
		"""
		label_counts = []

		label_count = adata[train_mask].obs[class_column].map(adata[train_mask].obs[class_column].value_counts())
		label_counts.append(label_count)

		label_counts = pd.concat(label_counts, ignore_index=True)
		label_counts = np.log(label_counts / 10 + 1)
		label_counts = 1 / label_counts

		sampling_weights = label_counts / sum(label_counts)

		return sampling_weights

	@abstractmethod
	def calculate_loss(self, scRNA_pred, scRNA, tcr_seq_pred, tcr_seq, loss_weights, scRNA_criterion, TCR_criterion,
					   size_factor):
		raise NotImplementedError

	def calculate_rna_loss(self, prediction, truth, loss_function):
		loss = loss_function(prediction, truth)
		return loss

	def calculate_tcr_annealing(self, init_tcr_loss_weight, n_epochs, epoch):
		if epoch > n_epochs / 2:
			return init_tcr_loss_weight * (epoch - n_epochs / 2) / (n_epochs / 2)
		else:
			return 0.0

	@abstractmethod
	def calculate_kld_loss(self, loss_weights, mu, logvar, epoch):
		"""
		Calculate the kld loss and z depending on the model type
		:param loss_weights: list, containing the weightening of the different loss factors
		:param mu: mean of the VAE latent space
		:param logvar: log(variance) of the VAE latent space
		:param epoch: current epoch as integer
		:param kl_annealing_epochs: amount of kl annealing epochs
		:return:
		"""
		raise NotImplementedError('Implement this in the different model versions')

# todo move this to model
"""
else (no poe, no moe):
kld_loss = loss_weights[2] * kl_criterion(mu, logvar) * self.kl_annealing(e, kl_annealing_epochs)
z = mu  # make z deterministic by using the mean
return kld_loss, z
"""