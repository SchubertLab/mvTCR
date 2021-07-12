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
from .losses.CoVWeightening import CoVWeighter

from .base_model import BaseModel
from tcr_embedding.evaluation.Imputation import run_imputation_evaluation
from tcr_embedding.evaluation.WrapperFunctions import get_model_prediction_function
from tcr_embedding.models.scGen import PertubationPredictor


class VAEBaseModel(BaseModel, ABC):
	def __init__(self,
				 adatas,  # adatas containing gene expression and TCR-seq
				 aa_to_id,
				 seq_model_arch,  # seq model architecture
				 seq_model_hyperparams,  # dict of seq model hyperparameters
				 scRNA_model_arch,
				 scRNA_model_hyperparams,
				 zdim,  # zdim
				 hdim,  # hidden dimension of encoder for each modality
				 activation,
				 dropout,
				 batch_norm,
				 shared_hidden=[],
				 names=[],
				 gene_layers=[],
				 seq_keys=[],
				 params_additional=None,
				 conditional=None,
				 rna_priority=None,
				 ):
		"""
		VAE Base Model, used for both single and joint models
		:param adatas: list of adatas containing train and val set
		:param names: list of str names for each adata, same order as adatas
		:param aa_to_id: dict containing mapping from amino acid symbol to label idx
		:param seq_model_arch: str name of TCR model architecture, currently supports ['BiGRU', 'CNN', 'Transformer']
		:param seq_model_hyperparams:  dict of hyperparameters used by the TCR model
		:param scRNA_model_arch: str name of scRNA model architecture, currently supports ['MLP']
		:param scRNA_model_hyperparams: dict of hyperparameters used by the scRNA model
		:param zdim: int, dimension of zdim
		:param hdim: int, dimension of shared dimension, i.e. output dim of scRNA and TCR model
		:param activation: str activation of shared encoder and decoder options ['relu', 'leakyrelu', 'linear']
		:param dropout: float dropout probability of shared encoder and decoder
		:param batch_norm: bool, use batch norm in shared encoder and decoder
		:param shared_hidden: list of ints or [], hidden layer dimension of shared encoder and decoder, can be empty for only a linear layer
		:param gene_layers: list of str or [], keys for scRNA data, i.e. adata.layer[gene_layers[i]] for each dataset i, or empty to use adata.X
		:param seq_keys: list of str or [], keys for TCR data, i.e. adata.obsm[seq_keys[i]] for each dataset i, or empty to use adata.obsm['tcr_seq']
		:param conditional: str or None, if None a normal VAE is used, if str then the str determines the adata.obsm[conditional] as conditioning variable
		:param rna_priority: int or None, if not None, the TCR module is only trained every n-th iteration
		"""

		assert len(adatas) == len(names)
		assert len(adatas) == len(seq_keys) or len(seq_keys) == 0
		assert len(adatas) == len(gene_layers) or len(gene_layers) == 0

		self.adatas = adatas
		if len(names) == 0:
			names = [f'dataset_{i}' for i in range(len(adatas))]
		self.names = names

		self._train_history = defaultdict(list)
		self._val_history = defaultdict(list)
		self.seq_model_arch = seq_model_arch
		self.conditional = conditional
		self.rna_priority = rna_priority

		if 'max_tcr_length' not in seq_model_hyperparams:
			seq_model_hyperparams['max_tcr_length'] = self.get_max_tcr_length()

		self.params = {'seq_model_arch': seq_model_arch, 'seq_model_hyperparams': seq_model_hyperparams,
					   'scRNA_model_arch': scRNA_model_arch, 'scRNA_model_hyperparams': scRNA_model_hyperparams,
					   'zdim': zdim, 'hdim': hdim, 'activation': activation, 'dropout': dropout,
					   'batch_norm': batch_norm,
					   'shared_hidden': shared_hidden}
		self.aa_to_id = aa_to_id

		if len(seq_keys) == 0:
			self.seq_keys = ['tcr_seq'] * len(adatas)
		else:
			self.seq_keys = seq_keys

		if len(gene_layers) == 0:
			self.gene_layers = [None] * len(adatas)
		else:
			self.gene_layers = gene_layers

		self.model_type = 'unsupervised'

		# supervised specific attributes
		self.label_key = None  # used for supervised and semi-supervised model
		self.label_to_specificity = None
		self.moe = False
		self.poe = False

		self.params_additional = params_additional

		self.tcr_annealing = False
		if type(params_additional) == dict:
			if 'tcr_annealing' in params_additional:
				self.tcr_annealing = params_additional['tcr_annealing']

	def train(self,
			  experiment_name='example',
			  n_iters=None,
			  n_epochs=100,
			  batch_size=64,
			  lr=3e-4,
			  losses=None,
			  loss_weights=None,
			  kl_annealing_epochs=None,
			  val_split='set',
			  metadata=None,
			  early_stop=None,
			  balanced_sampling=None,
			  log_divisor=10,
			  validate_every=1,
			  save_every=1,
			  save_path='../saved_models/',
			  save_last_model=False,
			  num_workers=0,
			  continue_training=False,
			  device=None,
			  comet=None,
			  tune=None
			  ):
		"""
		Train the model for n_epochs
		:param experiment_name: Name of experiment, used to save the model weights
		:param n_iters: None or int, number of iterations if int it overwrites n_epochs
		:param n_epochs: None or int, number of epochs to train, if None n_iters needs to be int
		:param batch_size: int, batch_size
		:param lr: float, learning rate
		:param losses: list of str, losses[0] := Loss for scRNA reconstruction, losses[1] := Loss for TCR reconstruction
		:param loss_weights: list of floats, loss_weights[0]:=weight or scRNA loss, loss_weights[1]:=weight for TCR loss, loss_weights[2] := KLD Loss
		:param kl_annealing_epochs: int or None, int number of epochs until kl reaches maximum warmup, if None this value is set to 30% of n_epochs
		:param val_split: str or float, if str it indicates the adata.obs[val_split] containing 'train' and 'val', if float, adata is split randomly by val_split
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param early_stop: int, stop training after this number of epochs if val loss is not improving anymore
		:param balanced_sampling: None or str, indicate adata.obs column to balance
		:param validate_every: int, epochs to validate
		:param save_every: int, epochs to save intermediate model weights
		:param save_path: str, path to directory to save model
		:param num_workers: int, number of workers for dataloader
		:param continue_training: bool, continue training from previous state, loads last saved model with the same experiment name and optimizer state
		:param device: None or str, if None device is determined automatically, based on if cuda.is_available
		:param comet: None or comet_ml.Experiment object
		:return:
		"""

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if losses is None:
			losses = ['MSE', 'CE']

		# initialize Coefficient of Variance weighting method if required
		if 'CoV' in self.names[0]:
			cov_weighter = CoVWeighter(comet=comet, device=device)
		else:
			cov_weighter = None
		if metadata is None:
			metadata = []

		if balanced_sampling is not None and balanced_sampling not in metadata:
			metadata.append(balanced_sampling)
		print('Create Dataloader')
		# Initialize dataloader
		if continue_training:
			self.train_masks = self.load_data(os.path.join(save_path, f'{experiment_name}_last_model.pt'))
			train_datasets, val_datasets, _ = self.create_datasets(self.adatas, self.names, self.gene_layers,
																   self.seq_keys, val_split, metadata, self.train_masks,
																   self.label_key)
		else:
			train_datasets, val_datasets, self.train_masks = self.create_datasets(self.adatas, self.names,
																				  self.gene_layers, self.seq_keys,
																				  val_split, metadata,
																				  label_key=self.label_key)

		if balanced_sampling is None:
			train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers,
										  worker_init_fn=self.seed_worker)
		else:
			sampling_weights = self.calculate_sampling_weights(self.adatas, self.train_masks, self.names,
															   class_column=balanced_sampling, log_divisor=log_divisor)
			sampler = WeightedRandomSampler(weights=sampling_weights, num_samples=len(sampling_weights),
											replacement=True)
			# shuffle is mutually exclusive to sampler, but sampler is anyway shuffled
			if comet is not None:
				comet.log_parameters({'sampling_weight_min': sampling_weights.min(),
									  'sampling_weight_max': sampling_weights.max()})
			train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers,
										  sampler=sampler, worker_init_fn=self.seed_worker)
		val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers)
		print('Dataloader created')

		try:
			os.makedirs(save_path)  # Create directory to prevent Error while saving model weights
		except:
			pass

		# Initialize Loss and Optimizer
		assert len(losses) == 2, 'losses needs to contain two elements corresponding to scRNA and TCR-seq loss'

		if n_iters is not None:
			n_epochs = n_iters // len(train_dataloader)

		if kl_annealing_epochs is None:
			kl_annealing_epochs = int(0.3 * n_epochs)

		if loss_weights is None:
			loss_weights = [1.0] * 3
		elif len(loss_weights) == 3 or len(loss_weights) == 4:
			loss_weights = loss_weights
		else:
			raise ValueError(f'length of loss_weights must be 3, 4 (supervised) or None.')

		self.losses = losses
		if losses[0] == 'MSE':
			scRNA_criterion = nn.MSELoss()
		elif losses[0] == 'NB':
			scRNA_criterion = NB()
		else:
			raise ValueError(f'{losses[0]} loss is not implemented')

		if losses[1] == 'CE':
			TCR_criterion = nn.CrossEntropyLoss(ignore_index=self.aa_to_id['_'])
		else:
			raise ValueError(f'{losses[1]} loss is not implemented')

		KL_criterion = KLD()
		CLS_criterion = nn.CrossEntropyLoss()

		self.model = self.model.to(device)

		self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

		self.best_loss = 99999999999
		self.best_cls_metric = -1
		self.best_knn_metric = -1
		no_improvements = 0
		self.epoch = 0
		epoch2step = 256 / batch_size  # normalization factor of epoch -> step, as one epoch with different batch_size results in different numbers of iterations
		epoch2step *= 1000  # to avoid decimal points, as we multiply with a float number

		if continue_training:
			# Load model and optimizer state_dict, as well as epoch and history
			self.load(os.path.join(save_path, f'{experiment_name}_last_model.pt'))
			self.epoch += 1
			print(f'Continue training from epoch {self.epoch}')
		pbar = tqdm(range(self.epoch, n_epochs + 1), 'Epoch: ')

		init_tcr_loss_weight = loss_weights[1]

		iteration = torch.tensor(0)

		for e in pbar:
			self.epoch = e
			# TRAIN LOOP
			loss_train_total = []
			scRNA_loss_train_total = []
			TCR_loss_train_total = []
			KLD_loss_train_total = []
			cls_loss_train_total = []
			labels_gt_list = []
			labels_pred_list = []
			cov_TCR_weights = []
			cov_scRNA_weights = []

			self.model.train()
			for scRNA, tcr_seq, size_factor, name, index, seq_len, metadata_batch, labels, conditional in train_dataloader:
				scRNA = scRNA.to(device)
				tcr_seq = tcr_seq.to(device)
				iteration = iteration.to(device)

				if self.conditional is not None:
					conditional = conditional.to(device)
				else:
					conditional = None

				z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, tcr_seq, seq_len, conditional, iteration)
				iteration += 1

				if self.moe:
					KLD_loss = 0.5 * loss_weights[2] * \
							   (KL_criterion(mu[0], logvar[0]) + KL_criterion(mu[1], logvar[1])) * \
							   self.kl_annealing(e, kl_annealing_epochs)
					z = 0.5 * (mu[0] + mu[1])  # mean of latent space from both modalities for mmvae
				elif self.poe:
					KLD_loss = 1.0 / 3.0 * loss_weights[2] * self.kl_annealing(e, kl_annealing_epochs) * \
							   (KL_criterion(mu[0], logvar[0]) +
								KL_criterion(mu[1], logvar[1]) +
								KL_criterion(mu[2], logvar[2]))
					z = mu[2]  # use joint latent variable for further downstream tasks
				else:
					KLD_loss = loss_weights[2] * KL_criterion(mu, logvar) * self.kl_annealing(e, kl_annealing_epochs)
					z = mu  # make z deterministic by using the mean

				if self.tcr_annealing:
					loss_weights[1] = self.calculate_tcr_annealing(init_tcr_loss_weight, n_epochs, e)
					if comet is not None:
						comet.log_metric('tcr_loss_weight', loss_weights[1])

				loss, scRNA_loss, TCR_loss = self.calculate_loss(scRNA_pred, scRNA, tcr_seq_pred, tcr_seq, loss_weights,
																 scRNA_criterion, TCR_criterion, size_factor)
				if cov_weighter is not None:
					# first remove the loss weights from the individual losses
					weight_tcr, weight_rna = cov_weighter.get_weights(TCR_loss, scRNA_loss, is_train=True)
					scRNA_loss = scRNA_loss / (loss_weights[0] + 1e-8) * weight_rna
					TCR_loss = TCR_loss / (loss_weights[1] + 1e-8) * weight_tcr
					# second calculate new weights based on cov
					loss = TCR_loss + scRNA_loss
					cov_TCR_weights.append(weight_tcr)
					cov_scRNA_weights.append(weight_rna)

				loss = loss + KLD_loss
				if self.model_type == 'supervised':
					labels_pred = self.model.classify(z)
					labels = labels.to(device)
					cls_loss = loss_weights[3] * CLS_criterion(labels_pred, labels)
					loss += cls_loss
					labels_gt_list.append(labels)
					labels_pred_list.append(labels_pred.argmax(dim=-1))
				else:
					cls_loss = torch.FloatTensor([0])

				self.optimizer.zero_grad()
				loss.backward()
				# todo add scrna priority here
				self.optimizer.step()

				loss_train_total.append(loss.detach())
				scRNA_loss_train_total.append(scRNA_loss.detach())
				TCR_loss_train_total.append(TCR_loss.detach())
				KLD_loss_train_total.append(KLD_loss.detach())
				cls_loss_train_total.append(cls_loss)

			if e % validate_every == 0:
				loss_train_total = torch.stack(loss_train_total).mean().item()
				scRNA_loss_train_total = torch.stack(scRNA_loss_train_total).mean().item()
				TCR_loss_train_total = torch.stack(TCR_loss_train_total).mean().item()
				KLD_loss_train_total = torch.stack(KLD_loss_train_total).mean().item()
				cls_loss_train_total = torch.stack(cls_loss_train_total).mean().item()
				if len(cov_scRNA_weights) != 0:
					cov_scRNA_weights = torch.stack(cov_scRNA_weights).mean().item()
					cov_TCR_weights = torch.stack(cov_TCR_weights).mean().item()
					if comet is not None:
						comet.log_metrics({'CoV_scRNA_weight_norm': cov_scRNA_weights,
										   'CoV_TCR_weight_norm': cov_TCR_weights},
										  step=int(e * epoch2step), epoch=e)

				if comet is not None:
					comet.log_metrics({'Train Loss': loss_train_total,
									   'Train scRNA Loss': scRNA_loss_train_total,
									   'Train TCR Loss': TCR_loss_train_total,
									   'Train KLD Loss': KLD_loss_train_total,
									   'Train CLS Loss': cls_loss_train_total},
									  step=int(e * epoch2step), epoch=e)

				if self.model_type == 'supervised':  # and comet is not None:
					labels_gt_list = torch.cat(labels_gt_list).cpu().numpy()
					labels_gt_list = [self.label_to_specificity[x] for x in labels_gt_list]
					labels_pred_list = torch.cat(labels_pred_list).cpu().numpy()
					labels_pred_list = [self.label_to_specificity[x] for x in labels_pred_list]

					metrics = classification_report(labels_gt_list, labels_pred_list, output_dict=True)
					for antigen, metric in metrics.items():
						if antigen != 'accuracy':
							comet.log_metrics(metric, prefix=f'Train supervised {antigen}', step=int(e * epoch2step),
											  epoch=e)
						else:
							comet.log_metric('Train supervised accuracy', metric, step=int(e * epoch2step), epoch=e)

			# VALIDATION LOOP
			if e % validate_every == 0:
				with torch.no_grad():
					self.model.eval()
					loss_val_total = []
					scRNA_loss_val_total = []
					TCR_loss_val_total = []
					KLD_loss_val_total = []
					cls_loss_val_total = []
					labels_gt_list = []
					labels_pred_list = []

					for scRNA, tcr_seq, size_factor, name, index, seq_len, metadata_batch, labels, conditional in val_dataloader:
						scRNA = scRNA.to(device)
						tcr_seq = tcr_seq.to(device)
						if self.conditional is not None:
							conditional = conditional.to(device)
						else:
							conditional = None

						z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, tcr_seq, seq_len, conditional)

						if self.moe:
							KLD_loss = 0.5 * loss_weights[2] * \
									   (KL_criterion(mu[0], logvar[0]) + KL_criterion(mu[1], logvar[1])) * \
									   self.kl_annealing(e, kl_annealing_epochs)
							z = 0.5 * (mu[0] + mu[1])  # mean of latent space from both modalities for mmvae
						elif self.poe:
							KLD_loss = 1.0 / 3.0 * loss_weights[2] * self.kl_annealing(e, kl_annealing_epochs) * \
									   (KL_criterion(mu[0], logvar[0]) + KL_criterion(mu[1], logvar[1]) + KL_criterion(
										   mu[2], logvar[2]))
							z = z[2]  # use joint latent variable
						else:
							KLD_loss = loss_weights[2] * KL_criterion(mu, logvar) * self.kl_annealing(e,
																									  kl_annealing_epochs)

						loss, scRNA_loss, TCR_loss = self.calculate_loss(scRNA_pred, scRNA, tcr_seq_pred, tcr_seq,
																		 loss_weights, scRNA_criterion, TCR_criterion,
																		 size_factor)
						if cov_weighter is not None:
							# first remove the loss weights from the individual losses
							weight_tcr, weight_rna = cov_weighter.get_weights(TCR_loss, scRNA_loss, is_train=False)
							scRNA_loss = scRNA_loss / (loss_weights[0] + 1e-8) * weight_rna
							TCR_loss = TCR_loss / (loss_weights[1] + 1e-8) * weight_tcr
							# second calculate new weights based on cov
							loss = TCR_loss + scRNA_loss

						loss = loss + KLD_loss
						if self.model_type == 'supervised':
							labels_pred = self.model.classify(z)
							labels = labels.to(device)
							cls_loss = loss_weights[3] * CLS_criterion(labels_pred, labels)
							loss += cls_loss
							labels_gt_list.append(labels)
							labels_pred_list.append(labels_pred.argmax(dim=-1))
						else:
							cls_loss = torch.FloatTensor([0])

						loss_val_total.append(loss)
						scRNA_loss_val_total.append(scRNA_loss)
						TCR_loss_val_total.append(TCR_loss)
						KLD_loss_val_total.append(KLD_loss)
						cls_loss_val_total.append(cls_loss)

					self.model.train()

					loss_val_total = torch.stack(loss_val_total).mean().item()
					scRNA_loss_val_total = torch.stack(scRNA_loss_val_total).mean().item()
					TCR_loss_val_total = torch.stack(TCR_loss_val_total).mean().item()
					KLD_loss_val_total = torch.stack(KLD_loss_val_total).mean().item()
					cls_loss_val_total = torch.stack(cls_loss_val_total).mean().item()

					# Only for TCR annealing. Undo the loss weighting, and multiply TCR loss by 0.1 to account for different scales of RNA and TCR loss
					if (e > n_epochs / 2 and self.tcr_annealing):
						unweighted_rna_loss = scRNA_loss_val_total / loss_weights[0]
						unweighted_tcr_loss = TCR_loss_val_total / loss_weights[1]
						unweighted_rec_loss = unweighted_rna_loss + 0.1 * unweighted_tcr_loss
						if comet is not None:
							comet.log_metrics({'Val unweighted RNA Loss': unweighted_rna_loss,
											   'Val unweighted TCR Loss': unweighted_tcr_loss,
											   'Val unweighted total Loss': unweighted_rec_loss},
											  step=int(e * epoch2step), epoch=e)
					else:
						unweighted_rec_loss = 9999999999999

					if (loss_val_total < self.best_loss) and not self.tcr_annealing:
						self.best_loss = loss_val_total
						self.save(os.path.join(save_path, f'{experiment_name}_best_rec_model.pt'))
						no_improvements = 0
					elif (unweighted_rec_loss < self.best_loss) and self.tcr_annealing and (e > n_epochs / 2):
						self.best_loss = unweighted_rec_loss
						self.save(os.path.join(save_path, f'{experiment_name}_best_rec_model.pt'))
						no_improvements = 0
					else:
						no_improvements += validate_every

					# If TCR training due to TCR annealing didn't start yet or if KL warmup period didn't finish yet, reset the no_improvement counter
					if (self.tcr_annealing and e <= n_epochs / 2) or e < kl_annealing_epochs:
						no_improvements = 0

					if self.model_type == 'supervised' and comet is not None:
						labels_gt_list = torch.cat(labels_gt_list).cpu().numpy()
						labels_gt_list = [self.label_to_specificity[x] for x in labels_gt_list]
						labels_pred_list = torch.cat(labels_pred_list).cpu().numpy()
						labels_pred_list = [self.label_to_specificity[x] for x in labels_pred_list]

						metrics = classification_report(labels_gt_list, labels_pred_list, output_dict=True)
						for antigen, metric in metrics.items():
							if antigen != 'accuracy':
								comet.log_metrics(metric, prefix=f'Val supervised {antigen}',
												  step=int(e * epoch2step), epoch=e)
							else:
								comet.log_metric('Val supervised accuracy', metric, step=int(e * epoch2step), epoch=e)

						if metrics['weighted avg']['f1-score'] > self.best_cls_metric:
							self.best_cls_metric = metrics['weighted avg']['f1-score']
							self.save(os.path.join(save_path, f'{experiment_name}_best_cls_model.pt'))

						if tune is not None:
							tune.report(weighted_f1=metrics['weighted avg']['f1-score'])

					if comet is not None:
						comet.log_metrics({'Val Loss': loss_val_total,
										   'Val scRNA Loss': scRNA_loss_val_total,
										   'Val TCR Loss': TCR_loss_val_total,
										   'Val KLD Loss': KLD_loss_val_total,
										   'Val CLS Loss': cls_loss_val_total,
										   'Epochs without Improvements': no_improvements},
										  step=int(e * epoch2step), epoch=e)

			if e % validate_every == 0 and save_last_model:
				self.save(os.path.join(save_path, f'{experiment_name}_last_model.pt'))

			# kNN evaluation
			if save_every is not None and e % save_every == 0:
				if self.names[0] == '10x':
					self.report_validation_10x(batch_size, e, epoch2step, tune, comet, save_path, experiment_name, pbar)
				if 'reconstruction' in self.names[0]:
					self.report_reconstruction(loss_val_total, tune)
				if self.names[0] == 'scgen':
					self.report_scgen(tune, comet, e, epoch2step)

			if early_stop is not None and no_improvements > early_stop:
				print('Early stopped')
				break

			if torch.isnan(loss):
				print(f'Loss became NaN, Loss: {loss}')
				return

	def report_validation_10x(self, batch_size, epoch, epoch2step, tune, comet, save_path, experiment_name, pbar):
		"""
		Report the objective metric of the 10x dataset for hyper parameter optimization.
		:param batch_size: Batch size for creating the validation latent space
		:param epoch: epoch number for logging
		:param epoch2step: conversion parameter between epoch and steps
		:param tune: RayTune object to report to
		:param comet: Comet experiments for logging validation
		:param save_path: Path for saving trained models
		:param experiment_name: Name of the experiment for logging
		:return: Reports to RayTune log files and externally to comet, saves model.
		"""

		test_embedding_func = get_model_prediction_function(self, batch_size=batch_size)
		try:
			summary = run_imputation_evaluation(self.adatas[0], test_embedding_func, query_source='val',
												use_non_binder=True, use_reduced_binders=True)
		except:
			print(f'kNN did not work')
			if tune is not None:
				tune.report(weighted_f1=0.0)
			return

		metrics = summary['knn']
		if tune is not None:
			tune.report(weighted_f1=metrics['weighted avg']['f1-score'])

		if comet is not None:
			for antigen, metric in metrics.items():
				if antigen != 'accuracy':
					comet.log_metrics(metric, prefix=antigen, step=int(epoch * epoch2step), epoch=epoch)
				else:
					comet.log_metric('accuracy', metric, step=int(epoch * epoch2step), epoch=epoch)

		if metrics['weighted avg']['f1-score'] > self.best_knn_metric:
			self.best_knn_metric = metrics['weighted avg']['f1-score']
			self.save(os.path.join(save_path, f'{experiment_name}_best_knn_model.pt'))
			pbar.set_postfix(best_f1_score=self.best_knn_metric, best_epoch=epoch)

	def report_reconstruction(self, validation_loss, tune):
		"""
		Report the reconstruction loss as metric for hyper parameter optimization.
		:param validation_loss: Reconstruction loss on the validation set
		:param tune: Ray Tune experiment
		:return: Reports to RayTune files
		"""
		if tune is not None:
			tune.report(reconstruction=validation_loss)

	def report_scgen(self, tune, comet, epoch, epoch2step):
		evaluator = PertubationPredictor(self, self.adatas[0], verbosity=0)
		summary = evaluator.evaluate_pertubation(self.params_additional['pertubation'], {'set': 'val'},
												 self.params_additional['per_column'],
												 self.params_additional['indicator'])

		score = summary['top_100_genes']['r_squared']

		if tune is not None:
			tune.report(scgen=score)

		if comet is not None:
			comet.log_metrics(summary['all_genes'], prefix='all_genes',
							  step=int(epoch * epoch2step), epoch=epoch)
			comet.log_metrics(summary['top_100_genes'], prefix='top_100_genes',
							  step=int(epoch * epoch2step), epoch=epoch)
			try:
				for name, summary in summary['per_cluster'].items():
					comet.log_metrics(summary, prefix=f'all_genes_{name}',
									  step=int(epoch * epoch2step), epoch=epoch)
				for name, summary in summary['per_cluster_top_100'].items():
					comet.log_metrics(summary, prefix=f'top_100_genes_{name}',
									  step=int(epoch * epoch2step), epoch=epoch)
			except KeyError:
				pass

	def get_latent(self, adatas, batch_size=256, num_workers=0, names=[], gene_layers=[], seq_keys=[], metadata=[],
				   device=None, return_mean=True):
		"""
		Get latent
		:param adatas: list of adatas
		:param batch_size: int, batch size
		:param names: list of str names for each adata, same order as adatas
		:param num_workers: int, num_workers for dataloader
		:param gene_layers: list of str or [], keys for scRNA data, i.e. adata.layer[gene_layers[i]] for each dataset i, or empty to use adata.X
		:param seq_keys: list of str or [], keys for TCR data, i.e. adata.obsm[seq_keys[i]] for each dataset i, or empty to use adata.obsm['tcr_seq']
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param device: None or str, if None device is determined automatically, based on if cuda.is_available
		:param return_mean: bool, calculate latent space without sampling
		:return: adata containing embedding vector in adata.X for each cell and the specified metadata in adata.obs
		"""
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		if len(names) == 0:
			names = [f'dataset_{i}' for i in range(len(adatas))]

		if len(seq_keys) == 0:
			seq_keys = ['tcr_seq'] * len(adatas)
		else:
			seq_keys = seq_keys

		if len(gene_layers) == 0:
			gene_layers = [None] * len(adatas)
		else:
			gene_layers = gene_layers

		pred_datasets, _, _ = self.create_datasets(adatas, names, gene_layers, seq_keys, val_split=0, metadata=metadata)
		pred_dataloader = DataLoader(pred_datasets, batch_size=batch_size, shuffle=False, collate_fn=None,
									 num_workers=num_workers)

		zs = []
		with torch.no_grad():
			self.model = self.model.to(device)
			self.model.eval()
			for scRNA, tcr_seq, size_factor, name, index, seq_len, metadata_batch, labels, conditional in pred_dataloader:
				scRNA = scRNA.to(device)
				tcr_seq = tcr_seq.to(device)
				if self.conditional is not None:
					conditional = conditional.to(device)
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

	def predict_sc_rna(self, data_in):
		if not self.poe:
			joint_encoding = self.model.model.shared_decoder(data_in)
			prediction_cells = self.model.model.gene_decoder(joint_encoding).detach().cpu().numpy()
			return prediction_cells
		else:
			pass

	def create_datasets(self, adatas, names, layers, seq_keys, val_split, metadata=[], train_masks=None,
						label_key=None):
		"""
		Create torch Dataset, see above for the input
		:param adatas: list of adatas
		:param names: list of str
		:param layers:
		:param seq_keys:
		:param val_split:
		:param metadata:
		:param train_masks: None or list of train_masks: if None new train_masks are created, else the train_masks are used, useful for continuing training
		:return: train_dataset, val_dataset, train_masks (for continuing training)
		"""
		dataset_names_train = []
		dataset_names_val = []
		scRNA_datas_train = []
		scRNA_datas_val = []
		seq_datas_train = []
		seq_datas_val = []
		seq_len_train = []
		seq_len_val = []
		adatas_train = {}
		adatas_val = {}
		index_train = []
		index_val = []
		metadata_train = []
		metadata_val = []
		conditional_train = []
		conditional_val = []

		if train_masks is None:
			masks_exist = False
			train_masks = {}
		else:
			masks_exist = True

		# Iterates through datasets with corresponding dataset name, scRNA layer and TCR column key
		# Splits everything into train and val
		for i, (name, adata, layer, seq_key) in enumerate(zip(names, adatas, layers, seq_keys)):
			if masks_exist:
				train_mask = train_masks[name]
			else:
				if type(val_split) == str:
					train_mask = (adata.obs[val_split] == 'train').values
				else:
					# Create train mask for each dataset separately
					num_samples = adata.X.shape[0] if layer is None else len(adata.layers[layer].shape[0])
					train_mask = np.zeros(num_samples, dtype=np.bool)
					train_size = int(num_samples * (1 - val_split))
					if val_split != 0:
						train_mask[:train_size] = 1
					else:
						train_mask[:] = 1
					np.random.shuffle(train_mask)
				train_masks[name] = train_mask

			# Save dataset splits
			scRNA_datas_train.append(adata.X[train_mask] if layer is None else adata.layers[layer][train_mask])
			scRNA_datas_val.append(adata.X[~train_mask] if layer is None else adata.layers[layer][~train_mask])

			seq_datas_train.append(adata.obsm[seq_key][train_mask])
			seq_datas_val.append(adata.obsm[seq_key][~train_mask])

			seq_len_train += adata.obs['seq_len'][train_mask].to_list()
			seq_len_val += adata.obs['seq_len'][~train_mask].to_list()

			adatas_train[name] = adata[train_mask]
			adatas_val[name] = adata[~train_mask]

			dataset_names_train += [name] * adata[train_mask].shape[0]
			dataset_names_val += [name] * adata[~train_mask].shape[0]

			index_train += adata[train_mask].obs.index.to_list()
			index_val += adata[~train_mask].obs.index.to_list()

			metadata_train.append(adata.obs[metadata][train_mask].values)
			metadata_val.append(adata.obs[metadata][~train_mask].values)

			if self.conditional is not None:
				conditional_train.append(adata.obsm[self.conditional][train_mask])
				conditional_val.append(adata.obsm[self.conditional][~train_mask])
			else:
				conditional_train = None
				conditional_val = None

		train_dataset = TCRDataset(scRNA_datas_train, seq_datas_train, seq_len_train, adatas_train, dataset_names_train,
								   index_train, metadata_train, labels=None, conditional=conditional_train)
		val_dataset = TCRDataset(scRNA_datas_val, seq_datas_val, seq_len_val, adatas_val, dataset_names_val, index_val,
								 metadata_val, labels=None, conditional=conditional_val)

		return train_dataset, val_dataset, train_masks

	def calculate_sampling_weights(self, adatas, train_masks, names, class_column, log_divisor=10):
		"""
		Calculate sampling weights for more balanced sampling in case of imbalanced classes,
		:params class_column: str, key for class to be balanced
		:params log_divisor: divide the label counts by this factor before taking the log, higher number makes the sampling more uniformly balanced
		:return: list of weights
		"""
		label_counts = []
		for adata, name in zip(adatas, names):
			train_mask = train_masks[name]
			label_count = adata[train_mask].obs[class_column].map(adata[train_mask].obs[class_column].value_counts())
			label_counts.append(label_count)

		label_counts = pd.concat(label_counts, ignore_index=True)
		label_counts = np.log(label_counts / log_divisor + 1)
		label_counts = 1 / label_counts

		sampling_weights = label_counts / sum(label_counts)

		return sampling_weights

	@abstractmethod
	def calculate_loss(self, scRNA_pred, scRNA, tcr_seq_pred, tcr_seq, loss_weights, scRNA_criterion, TCR_criterion,
					   size_factor):
		raise NotImplementedError

	def calc_scRNA_rec_loss(self, scRNA_pred, scRNA, scRNA_criterion, size_factor=None, rec_loss_type='MSE'):
		if rec_loss_type == 'MSE':
			scRNA_loss_unweighted = scRNA_criterion(scRNA_pred, scRNA)
		elif rec_loss_type == 'NB':
			scRNA_pred = F.softmax(scRNA_pred, dim=-1)
			size_factor_view = size_factor.unsqueeze(1).expand(-1, scRNA_pred.shape[1]).to(scRNA_pred.device)
			dec_mean = scRNA_pred * size_factor_view
			dispersion = self.model.theta.T
			dispersion = torch.exp(dispersion)
			scRNA_loss_unweighted = - scRNA_criterion(scRNA_pred, dec_mean,
													  dispersion) * 1000  # TODO A Test, maybe delete afterwards
		else:
			raise NotImplementedError(f'{rec_loss_type} is not implemented')

		return scRNA_loss_unweighted

	def calculate_tcr_annealing(self, init_tcr_loss_weight, n_epochs, e):
		if e > n_epochs / 2:
			return init_tcr_loss_weight * (e - n_epochs / 2) / (n_epochs / 2)
		else:
			return 0.0
