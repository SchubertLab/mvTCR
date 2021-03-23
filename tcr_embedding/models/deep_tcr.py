import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier


from .losses.kld import KLD
from tcr_embedding.datasets.scdataset import DeepTCRDataset, TCRDataset
from tcr_embedding.models.cnn import CNNEncoder, CNNDecoder


class SharedEncoder(nn.Module):
	def __init__(self, xdim, zdim):
		super(SharedEncoder, self).__init__()

		self.layer1 = nn.Linear(xdim, 256)
		self.layer2 = nn.Linear(256, 256)

		self.z_mean = nn.Linear(256, zdim)
		self.z_logvar = nn.Linear(256, zdim)
		self.softplus = nn.Softplus()

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)

		z_mean = self.z_mean(x)
		z_logvar = self.z_logvar(x)
		z_logvar = self.softplus(z_logvar)
		return z_mean, z_logvar


class SharedDecoder(nn.Module):
	def __init__(self, zdim):
		super(SharedDecoder, self).__init__()
		self.layer1 = nn.Linear(zdim, 128)
		self.layer2 = nn.Linear(128, 256)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		return x


class VDJEncoder(nn.Module):
	def __init__(self, vdj_embedding_dim, num_v_alpha, num_j_alpha, num_v_beta, num_d_beta, num_j_beta):
		super(VDJEncoder, self).__init__()
		self.embedding_v_alpha = nn.Embedding(num_embeddings=num_v_alpha, embedding_dim=vdj_embedding_dim)
		self.embedding_j_alpha = nn.Embedding(num_embeddings=num_j_alpha, embedding_dim=vdj_embedding_dim)

		self.embedding_v_beta = nn.Embedding(num_embeddings=num_v_beta, embedding_dim=vdj_embedding_dim)
		self.embedding_d_beta = nn.Embedding(num_embeddings=num_d_beta, embedding_dim=vdj_embedding_dim)
		self.embedding_j_beta = nn.Embedding(num_embeddings=num_j_beta, embedding_dim=vdj_embedding_dim)

	def forward(self, x):
		v_alpha = self.embedding_v_alpha(x[:, 0])
		j_alpha = self.embedding_j_alpha(x[:, 1])
		v_beta = self.embedding_v_beta(x[:, 2])
		d_beta = self.embedding_d_beta(x[:, 3])
		j_beta = self.embedding_j_beta(x[:, 4])

		return v_alpha, j_alpha, v_beta, d_beta, j_beta


class VDJDecoder(nn.Module):
	def __init__(self, hdim, vdj_embedding_dim, num_v_alpha, num_j_alpha, num_v_beta, num_d_beta, num_j_beta):
		super(VDJDecoder, self).__init__()

		self.decoder_v_alpha = self.build_decoder(hdim, vdj_embedding_dim, num_v_alpha)
		self.decoder_j_alpha = self.build_decoder(hdim, vdj_embedding_dim, num_j_alpha)

		self.decoder_v_beta = self.build_decoder(hdim, vdj_embedding_dim, num_v_beta)
		self.decoder_d_beta = self.build_decoder(hdim, vdj_embedding_dim, num_d_beta)
		self.decoder_j_beta = self.build_decoder(hdim, vdj_embedding_dim, num_j_beta)

	def build_decoder(self, zdim, vdj_embedding_dim, num_genes):
		decoder = nn.Sequential(
			nn.Linear(zdim, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, vdj_embedding_dim),
			nn.ReLU(inplace=True),
			nn.Linear(vdj_embedding_dim, num_genes)
		)
		return decoder

	def forward(self, x):
		v_alpha = self.decoder_v_alpha(x)
		j_alpha = self.decoder_j_alpha(x)
		v_beta = self.decoder_v_beta(x)
		d_beta = self.decoder_d_beta(x)
		j_beta = self.decoder_j_beta(x)

		return v_alpha, j_alpha, v_beta, d_beta, j_beta


class DeepTCRTorch(nn.Module):
	def __init__(self, num_v_alpha, num_j_alpha, num_v_beta, num_d_beta, num_j_beta, num_seq_labels, zdim, seq_model_hyperparams, vdj_embedding_dim=48):
		super(DeepTCRTorch, self).__init__()

		self.alpha_encoder = CNNEncoder(seq_model_hyperparams, None, num_seq_labels, use_output_layer=False)
		self.beta_encoder = CNNEncoder(seq_model_hyperparams, None, num_seq_labels, use_output_layer=False)
		self.alpha_decoder = CNNDecoder(seq_model_hyperparams, zdim, num_seq_labels)
		self.beta_decoder = CNNDecoder(seq_model_hyperparams, zdim, num_seq_labels)

		self.vdj_encoder = VDJEncoder(vdj_embedding_dim, num_v_alpha, num_j_alpha, num_v_beta, num_d_beta, num_j_beta)
		self.vdj_decoder = VDJDecoder(256, vdj_embedding_dim, num_v_alpha, num_j_alpha, num_v_beta, num_d_beta, num_j_beta)

		self.shared_encoder = SharedEncoder((self.alpha_encoder.output_dim*2)+(vdj_embedding_dim*5), zdim=zdim)
		self.shared_decoder = SharedDecoder(zdim)

	def forward(self, alpha_seq, beta_seq, vdj):
		"""
		Forward pass of autoencoder
		:param alpha_seq: torch.LongTensor shape=[batch_size, seq_len]
		:param beta_seq: torch.LongTensor shape=[batch_size, seq_len]
		:param vdj: torch.LongTensor shape=[batch_size, 5]
		:return: alpha_pred, beta_pred, vdj_pred
		"""

		h_v_alpha, h_j_alpha, h_v_beta, h_d_beta, h_j_beta = self.vdj_encoder(vdj)  # shape=[batch_size, embedding_dim]
		h_alpha_seq = self.alpha_encoder(alpha_seq, None)  # shape=[batch_size, hdim]
		h_beta_seq = self.beta_encoder(beta_seq, None)  # shape=[batch_size, hdim]

		joint_feature = torch.cat([h_v_alpha, h_j_alpha, h_v_beta, h_d_beta, h_j_beta, h_alpha_seq, h_beta_seq], dim=-1)
		mu, logvar = self.shared_encoder(joint_feature)  # shape=[batch_size, zdim*2]
		z = self.reparameterize(mu, logvar)  # shape=[batch_size, zdim]
		joint_dec_feature = self.shared_decoder(z)  # shape=[batch_size, hdim*2]

		v_alpha, j_alpha, v_beta, d_beta, j_beta = self.vdj_decoder(joint_dec_feature)  # shape=[batch_size, num_vdj_genes]
		alpha_seq_pred = self.alpha_decoder(joint_dec_feature, None)
		beta_seq_pred = self.beta_decoder(joint_dec_feature, None)

		return z, mu, logvar, v_alpha, j_alpha, v_beta, d_beta, j_beta, alpha_seq_pred, beta_seq_pred

	def reparameterize(self, mu, log_var):
		"""
		https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
		:param mu: mean from the encoder's latent space
		:param log_var: log variance from the encoder's latent space
		"""
		std = torch.exp(0.5 * log_var)  # standard deviation
		eps = torch.randn_like(std)  # `randn_like` as we need the same size
		z = mu + (eps * std)  # sampling as if coming from the input space
		return z


class DeepTCR:
	def __init__(self,
				 adatas,  # adatas containing gene expression and TCR-seq
				 seq_model_hyperparams,  # dict of seq model hyperparameters
				 zdim=256,  # zdim
				 names=[],
				 ):
		"""
		VAE Base Model, used for both single and joint models
		:param adatas: list of adatas containing train and val set
		:param seq_model_hyperparams:  dict of hyperparameters used by the TCR model
		:param zdim: int, dimension of zdim
		"""

		self.adatas = adatas
		self.aa_to_id = adatas[0].uns['aa_to_id']
		self.v_alpha_to_id = adatas[0].uns['v_alpha_to_id']
		self.j_alpha_to_id = adatas[0].uns['j_alpha_to_id']
		self.v_beta_to_id = adatas[0].uns['v_beta_to_id']
		self.d_beta_to_id = adatas[0].uns['d_beta_to_id']
		self.j_beta_to_id = adatas[0].uns['j_beta_to_id']

		self._train_history = defaultdict(list)
		self._val_history = defaultdict(list)

		seq_model_hyperparams['max_tcr_length'] = 40
		self.params = {'seq_model_hyperparams': seq_model_hyperparams, 'zdim': zdim}

		if len(names) == 0:
			names = [f'dataset_{i}' for i in range(len(adatas))]
		self.names = names

		num_seq_labels = len(self.aa_to_id)
		num_v_alpha = len(self.v_alpha_to_id)
		num_j_alpha = len(self.j_alpha_to_id)
		num_v_beta = len(self.v_beta_to_id)
		num_d_beta = len(self.d_beta_to_id)
		num_j_beta = len(self.j_beta_to_id)

		self.model = DeepTCRTorch(num_v_alpha, num_j_alpha, num_v_beta, num_d_beta, num_j_beta, num_seq_labels, zdim, seq_model_hyperparams)

	def train(self,
			  experiment_name='example',
			  n_iters=None,
			  n_epochs=100,
			  batch_size=64,
			  lr=3e-4,
			  loss_weights=[],
			  val_split='set',
			  metadata=[],
			  early_stop=None,
			  validate_every=10,
			  save_every=100,
			  save_path='../saved_models/',
			  num_workers=0,
			  verbose=1,
			  continue_training=False,
			  device=None,
			  comet=None
			  ):
		"""
		Train the model for n_epochs
		:param experiment_name: Name of experiment, used to save the model weights
		:param n_iters: None or int, number of iterations if int it overwrites n_epochs
		:param n_epochs: None or int, number of epochs to train, if None n_iters needs to be int
		:param batch_size: int, batch_size
		:param lr: float, learning rate
		:param loss_weights: list of floats, loss_weights[0]:=weight or scRNA loss, loss_weights[1]:=weight for TCR loss, loss_weights[2] := KLD Loss
		:param val_split: str or float, if str it indicates the adata.obs[val_split] containing 'train' and 'val', if float, adata is split randomly by val_split
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment #TODO maybe delete this
		:param early_stop: int, stop training after this number of epochs if val loss is not improving anymore
		:param validate_every: int, epochs to validate
		:param save_every: int, epochs to save intermediate model weights
		:param save_path: str, path to directory to save model
		:param num_workers: int, number of workers for dataloader
		:param verbose: 0, 1 or 2 - 0: only tqdm progress bar, 1: include val metrics, 2: include train metrics
		:param continue_training: bool, continue training from previous state, loads last saved model and optimizer state
		:param device: None or str, if None device is determined automatically, based on if cuda.is_available
		:param comet: None or comet_ml.Experiment object
		:return:
		"""

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		print('Create Dataloader')
		# Initialize dataloader
		if continue_training:
			# train_masks = self.load_data(f'../saved_models/{experiment_name}_data.pt')
			self.train_masks = self.load_data(os.path.join(save_path, f'{experiment_name}_last_model.pt'))
			train_datasets, val_datasets, _ = self.create_datasets(self.adatas, self.names, val_split, metadata, self.train_masks)
		else:
			train_datasets, val_datasets, self.train_masks = self.create_datasets(self.adatas, self.names, val_split, metadata)

		train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, collate_fn=None, num_workers=num_workers)
		val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False, collate_fn=None, num_workers=num_workers)
		print('Dataloader created')

		try:
			os.makedirs(save_path)  # Create directory to prevent Error while saving model weights
		except:
			pass

		if n_iters is not None:
			n_epochs = n_iters // len(train_dataloader)

		if len(loss_weights) == 0:
			loss_weights = [1.0] * 3
		elif len(loss_weights) == 3:
			loss_weights = loss_weights
		else:
			raise ValueError(f'length of loss_weights must be 3 or [].')

		recon_criterion = nn.CrossEntropyLoss(ignore_index=self.aa_to_id['_'])  # DeepTCR doesn't ignore idx 0
		KL_criterion = KLD()

		self.model = self.model.to(device)
		self.model.train()

		self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

		self.best_loss = 99999999999
		no_improvements = 0
		self.epoch = 0
		epoch2step = 256 / batch_size  # normalization factor of epoch -> step, as one epoch with different batch_size results in different numbers of iterations
		epoch2step *= 1000  # to avoid decimal points, as we multiply with a float number

		if continue_training:
			# Load model and optimizer state_dict, as well as epoch and history
			self.load(os.path.join(save_path, f'{experiment_name}_last_model.pt'))
			self.epoch += 1
			print(f'Continue training from epoch {self.epoch}')

		for e in tqdm(range(self.epoch, n_epochs+1), 'Epoch: '):
			self.epoch = e
			# TRAIN LOOP
			loss_train_total = []
			alpha_seq_loss_total = []
			beta_seq_loss_total = []
			v_alpha_loss_total = []
			j_alpha_loss_total = []
			v_beta_loss_total = []
			d_beta_loss_total = []
			j_beta_loss_total = []
			KLD_loss_train_total = []
			# vdj, vdj[:, 0] = v_alpha, vdj[:, 1] = j_alpha, vdj[:, 2] = v_beta, vdj[:, 3] = d_beta, vdj[:, 4] = j_beta
			for alpha_seq, beta_seq, vdj, name, index, metadata_batch in train_dataloader:
				alpha_seq = alpha_seq.to(device)
				beta_seq = beta_seq.to(device)
				vdj = vdj.to(device)

				z, mu, logvar, v_alpha_p, j_alpha_p, v_beta_p, d_beta_p, j_beta_p, alpha_seq_p, beta_seq_p = self.model(alpha_seq, beta_seq, vdj)

				KLD_loss = loss_weights[2] * KL_criterion(mu, logvar)
				alpha_seq_loss = loss_weights[1] * recon_criterion(alpha_seq_p.flatten(end_dim=1), alpha_seq.flatten())
				beta_seq_loss = loss_weights[1] * recon_criterion(beta_seq_p.flatten(end_dim=1), beta_seq.flatten())

				v_alpha_loss = loss_weights[0] * recon_criterion(v_alpha_p, vdj[:, 0])
				j_alpha_loss = loss_weights[0] * recon_criterion(j_alpha_p, vdj[:, 1])

				v_beta_loss = loss_weights[0] * recon_criterion(v_beta_p, vdj[:, 2])
				d_beta_loss = loss_weights[0] * recon_criterion(d_beta_p, vdj[:, 3])
				j_beta_loss = loss_weights[0] * recon_criterion(j_beta_p, vdj[:, 4])

				loss = alpha_seq_loss + beta_seq_loss + v_alpha_loss + j_alpha_loss + v_beta_loss + d_beta_loss + j_beta_loss + KLD_loss

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				loss_train_total.append(loss.detach())
				alpha_seq_loss_total.append(alpha_seq_loss.detach())
				beta_seq_loss_total.append(beta_seq_loss.detach())
				v_alpha_loss_total.append(v_alpha_loss.detach())
				j_alpha_loss_total.append(j_alpha_loss.detach())
				v_beta_loss_total.append(v_beta_loss.detach())
				d_beta_loss_total.append(d_beta_loss.detach())
				j_beta_loss_total.append(j_beta_loss.detach())
				KLD_loss_train_total.append(KLD_loss.detach())

			if e % validate_every == 0:
				# Save train losses
				loss_train_total = torch.stack(loss_train_total).mean().item()
				alpha_seq_loss_total = torch.stack(alpha_seq_loss_total).mean().item()
				beta_seq_loss_total = torch.stack(beta_seq_loss_total).mean().item()
				v_alpha_loss_total = torch.stack(v_alpha_loss_total).mean().item()
				j_alpha_loss_total = torch.stack(j_alpha_loss_total).mean().item()
				v_beta_loss_total = torch.stack(v_beta_loss_total).mean().item()
				d_beta_loss_total = torch.stack(d_beta_loss_total).mean().item()
				j_beta_loss_total = torch.stack(j_beta_loss_total).mean().item()
				KLD_loss_train_total = torch.stack(KLD_loss_train_total).mean().item()

				self._train_history['epoch'].append(e)
				self._train_history['total_loss'].append(loss_train_total)
				self._train_history['alpha_seq_loss'].append(alpha_seq_loss_total)
				self._train_history['beta_seq_loss'].append(beta_seq_loss_total)
				self._train_history['v_alpha_loss'].append(v_alpha_loss_total)
				self._train_history['j_alpha_loss'].append(j_alpha_loss_total)
				self._train_history['v_beta_loss'].append(v_beta_loss_total)
				self._train_history['d_beta_loss'].append(d_beta_loss_total)
				self._train_history['j_beta_loss'].append(j_beta_loss_total)
				self._train_history['KLD_loss'].append(KLD_loss_train_total)

				if verbose >= 2:
					print('\n')
					print(f'Train Loss: {loss_train_total}')
					print(f'Train Alpha Seq Loss: {alpha_seq_loss_total}')
					print(f'Train Beta Seq Loss: {beta_seq_loss_total}')
					print(f'Train V Alpha Loss: {v_alpha_loss_total}')
					print(f'Train J Alpha Loss: {j_alpha_loss_total}')
					print(f'Train V Beta Loss: {v_beta_loss_total}')
					print(f'Train D Beta Loss: {d_beta_loss_total}')
					print(f'Train J Beta Loss: {j_beta_loss_total}')
					print(f'Train KLD Loss: {KLD_loss_train_total}\n')
				if comet is not None:
					comet.log_metrics({'Train Loss': loss_train_total,
									   'Train Alpha Seq Loss': alpha_seq_loss_total,
									   'Train Beta Seq Loss': beta_seq_loss_total,
									   'Train V Alpha Loss': v_alpha_loss_total,
									   'Train J Alpha Loss': j_alpha_loss_total,
									   'Train V Beta Loss': v_beta_loss_total,
									   'Train D Beta Loss': d_beta_loss_total,
									   'Train J Beta Loss': j_beta_loss_total,
									   'Train KLD Loss': KLD_loss_train_total},
									  step=int(e*epoch2step), epoch=e)

				# VALIDATION LOOP
				with torch.no_grad():
					self.model.eval()
					loss_val_total = []
					alpha_seq_loss_total = []
					beta_seq_loss_total = []
					v_alpha_loss_total = []
					j_alpha_loss_total = []
					v_beta_loss_total = []
					d_beta_loss_total = []
					j_beta_loss_total = []
					KLD_loss_val_total = []

					for alpha_seq, beta_seq, vdj, name, index, metadata_batch in val_dataloader:
						alpha_seq = alpha_seq.to(device)
						beta_seq = beta_seq.to(device)
						vdj = vdj.to(device)

						z, mu, logvar, v_alpha_p, j_alpha_p, v_beta_p, d_beta_p, j_beta_p, alpha_seq_p, beta_seq_p = self.model(alpha_seq, beta_seq, vdj)

						KLD_loss = loss_weights[2] * KL_criterion(mu, logvar)
						alpha_seq_loss = loss_weights[1] * recon_criterion(alpha_seq_p.flatten(end_dim=1), alpha_seq.flatten())
						beta_seq_loss = loss_weights[1] * recon_criterion(beta_seq_p.flatten(end_dim=1), beta_seq.flatten())

						v_alpha_loss = loss_weights[0] * recon_criterion(v_alpha_p, vdj[:, 0])
						j_alpha_loss = loss_weights[0] * recon_criterion(j_alpha_p, vdj[:, 1])

						v_beta_loss = loss_weights[0] * recon_criterion(v_beta_p, vdj[:, 2])
						d_beta_loss = loss_weights[0] * recon_criterion(d_beta_p, vdj[:, 3])
						j_beta_loss = loss_weights[0] * recon_criterion(j_beta_p, vdj[:, 4])

						loss = alpha_seq_loss + beta_seq_loss + v_alpha_loss + j_alpha_loss + v_beta_loss + d_beta_loss + j_beta_loss + KLD_loss

						loss_val_total.append(loss.detach())
						alpha_seq_loss_total.append(alpha_seq_loss.detach())
						beta_seq_loss_total.append(beta_seq_loss.detach())
						v_alpha_loss_total.append(v_alpha_loss.detach())
						j_alpha_loss_total.append(j_alpha_loss.detach())
						v_beta_loss_total.append(v_beta_loss.detach())
						d_beta_loss_total.append(d_beta_loss.detach())
						j_beta_loss_total.append(j_beta_loss.detach())
						KLD_loss_val_total.append(KLD_loss.detach())

					self.model.train()

					loss_val_total = torch.stack(loss_val_total).mean().item()
					alpha_seq_loss_total = torch.stack(alpha_seq_loss_total).mean().item()
					beta_seq_loss_total = torch.stack(beta_seq_loss_total).mean().item()
					v_alpha_loss_total = torch.stack(v_alpha_loss_total).mean().item()
					j_alpha_loss_total = torch.stack(j_alpha_loss_total).mean().item()
					v_beta_loss_total = torch.stack(v_beta_loss_total).mean().item()
					d_beta_loss_total = torch.stack(d_beta_loss_total).mean().item()
					j_beta_loss_total = torch.stack(j_beta_loss_total).mean().item()
					KLD_loss_val_total = torch.stack(KLD_loss_val_total).mean().item()

					self._val_history['epoch'].append(e)
					self._val_history['total_loss'].append(loss_val_total)
					self._val_history['alpha_seq_loss'].append(alpha_seq_loss_total)
					self._val_history['beta_seq_loss'].append(beta_seq_loss_total)
					self._val_history['v_alpha_loss'].append(v_alpha_loss_total)
					self._val_history['j_alpha_loss'].append(j_alpha_loss_total)
					self._val_history['v_beta_loss'].append(v_beta_loss_total)
					self._val_history['d_beta_loss'].append(d_beta_loss_total)
					self._val_history['j_beta_loss'].append(j_beta_loss_total)
					self._val_history['KLD_loss'].append(KLD_loss_val_total)
					if verbose >= 1:
						print('\n')
						print(f'Val Loss: {loss_val_total}')
						print(f'Val Alpha Seq Loss: {alpha_seq_loss_total}')
						print(f'Val Beta Seq Loss: {beta_seq_loss_total}')
						print(f'Val V Alpha Loss: {v_alpha_loss_total}')
						print(f'Val J Alpha Loss: {j_alpha_loss_total}')
						print(f'Val V Beta Loss: {v_beta_loss_total}')
						print(f'Val D Beta Loss: {d_beta_loss_total}')
						print(f'Val J Beta Loss: {j_beta_loss_total}')
						print(f'Val KLD Loss: {KLD_loss_val_total}\n')

					if loss_val_total < self.best_loss:
						self.best_loss = loss_val_total
						self.save(os.path.join(save_path, f'{experiment_name}_best_model.pt'))
						no_improvements = 0
					# KL warmup periods is grace period

					if comet is not None:
						comet.log_metrics({'Val Loss': loss_val_total,
										   'Val Alpha Seq Loss': alpha_seq_loss_total,
										   'Val Beta Seq Loss': beta_seq_loss_total,
										   'Val V Alpha Loss': v_alpha_loss_total,
										   'Val J Alpha Loss': j_alpha_loss_total,
										   'Val V Beta Loss': v_beta_loss_total,
										   'Val D Beta Loss': d_beta_loss_total,
										   'Val J Beta Loss': j_beta_loss_total,
										   'Val KLD Loss': KLD_loss_val_total,
										   'Epochs without Improvements': no_improvements},
										  step=int(e * epoch2step), epoch=e)

				self.save(os.path.join(save_path, f'{experiment_name}_last_model.pt'))

			if save_every is not None and e % save_every == 0:
				self.save(os.path.join(save_path, f'{experiment_name}_epoch_{str(e).zfill(5)}.pt'))

			if early_stop is not None and no_improvements > early_stop:
				print('Early stopped')
				break

	def get_latent(self, adatas, batch_size=256, num_workers=0, names=[], metadata=[], device=None):
		"""
		Get latent
		:param adatas: list of adatas
		:param batch_size: int, batch size
		:param names: list of str names for each adata, same order as adatas
		:param num_workers: int, num_workers for dataloader
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param device: None or str, if None device is determined automatically, based on if cuda.is_available
		:return: adata containing embedding vector in adata.X for each cell and the specified metadata in adata.obs
		"""
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		if len(names) == 0:
			names = [f'dataset_{i}' for i in range(len(adatas))]

		pred_datasets, _, _ = self.create_datasets(adatas, names, val_split=0, metadata=metadata)
		pred_dataloader = DataLoader(pred_datasets, batch_size=batch_size, shuffle=False, collate_fn=None, num_workers=num_workers)

		zs = []
		with torch.no_grad():
			self.model = self.model.to(device)
			self.model.eval()
			for alpha_seq, beta_seq, vdj, name, index, metadata_batch in pred_dataloader:
				alpha_seq = alpha_seq.to(device)
				beta_seq = beta_seq.to(device)
				vdj = vdj.to(device)

				z, mu, logvar, v_alpha_p, j_alpha_p, v_beta_p, d_beta_p, j_beta_p, alpha_seq_p, beta_seq_p = self.model(alpha_seq, beta_seq, vdj)

				z = sc.AnnData(z.detach().cpu().numpy())
				z.obs['barcode'] = index
				z.obs['dataset'] = name
				z.obs[metadata] = np.array(metadata_batch).T
				zs.append(z)

		return sc.AnnData.concatenate(*zs)


	def create_datasets(self, adatas, names, val_split, metadata=[], train_masks=None):
		"""
		Create torch Dataset, see above for the input
		:param adatas: list of adatas
		:param names: list of str
		:param layers:
		:param val_split:
		:param metadata:
		:param train_masks: None or list of train_masks: if None new train_masks are created, else the train_masks are used, useful for continuing training
		:return: train_dataset, val_dataset, train_masks (for continuing training)
		"""
		dataset_names_train = []
		dataset_names_val = []
		alpha_datas_train = []
		alpha_datas_val = []
		beta_datas_train = []
		beta_datas_val = []
		adatas_train = {}
		adatas_val = {}
		index_train = []
		index_val = []
		vdj_train = {'v_alpha': [], 'j_alpha': [], 'v_beta': [], 'd_beta': [], 'j_beta': []}
		vdj_val = {'v_alpha': [], 'j_alpha': [], 'v_beta': [], 'd_beta': [], 'j_beta': []}
		metadata_train = []
		metadata_val = []

		if train_masks is None:
			masks_exist = False
			train_masks = {}
		else:
			masks_exist = True

		# Iterates through datasets with corresponding dataset name, scRNA layer and TCR column key
		# Splits everything into train and val
		for i, (name, adata) in enumerate(zip(names, adatas)):
			if masks_exist:
				train_mask = train_masks[name]
			else:
				if type(val_split) == str:
					train_mask = (adata.obs[val_split] == 'train').values
				else:
					# Create train mask for each dataset separately
					num_samples = adata.X.shape[0]
					train_mask = np.zeros(num_samples, dtype=np.bool)
					train_size = int(num_samples * (1 - val_split))
					if val_split != 0:
						train_mask[:train_size] = 1
					else:
						train_mask[:] = 1
					np.random.shuffle(train_mask)
				train_masks[name] = train_mask

			# Save dataset splits
			alpha_datas_train.append(adata.obsm['alpha_seq'][train_mask])
			alpha_datas_val.append(adata.obsm['alpha_seq'][~train_mask])

			beta_datas_train.append(adata.obsm['beta_seq'][train_mask])
			beta_datas_val.append(adata.obsm['beta_seq'][~train_mask])

			adatas_train[name] = adata[train_mask]
			adatas_val[name] = adata[~train_mask]

			dataset_names_train += [name] * adata[train_mask].shape[0]
			dataset_names_val += [name] * adata[~train_mask].shape[0]

			index_train += adata[train_mask].obs.index.to_list()
			index_val += adata[~train_mask].obs.index.to_list()

			vdj_train['v_alpha'].append(adata[train_mask].obsm['v_alpha'])
			vdj_train['j_alpha'].append(adata[train_mask].obsm['j_alpha'])
			vdj_train['v_beta'].append(adata[train_mask].obsm['v_beta'])
			vdj_train['d_beta'].append(adata[train_mask].obsm['d_beta'])
			vdj_train['j_beta'].append(adata[train_mask].obsm['j_beta'])

			vdj_val['v_alpha'].append(adata[~train_mask].obsm['v_alpha'])
			vdj_val['j_alpha'].append(adata[~train_mask].obsm['j_alpha'])
			vdj_val['v_beta'].append(adata[~train_mask].obsm['v_beta'])
			vdj_val['d_beta'].append(adata[~train_mask].obsm['d_beta'])
			vdj_val['j_beta'].append(adata[~train_mask].obsm['j_beta'])

			metadata_train.append(adata.obs[metadata][train_mask].values)
			metadata_val.append(adata.obs[metadata][~train_mask].values)

		train_dataset = DeepTCRDataset(alpha_datas_train, beta_datas_train, vdj_train, adatas_train, dataset_names_train, index_train, metadata_train)
		val_dataset = DeepTCRDataset(alpha_datas_val, beta_datas_val, vdj_val, adatas_val, dataset_names_val, index_val, metadata_val)

		return train_dataset, val_dataset, train_masks

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
					  'train_masks': self.train_masks}
		try:
			model_file['optimizer'] = self.optimizer.state_dict()
		except:
			pass
		torch.save(model_file, filepath)

	def load(self, filepath):
		""" Load model and optimizer state, and auxiliary data for continuing training """
		model_file = torch.load(os.path.join(filepath))  # , map_location=self.device)
		self.model.load_state_dict(model_file['state_dict'])
		self._train_history = model_file['train_history']
		self._val_history = model_file['val_history']
		self.epoch = model_file['epoch']
		self.aa_to_id = model_file['aa_to_id']
		self.best_loss = model_file['best_loss']
		self.train_masks = model_file['train_masks']
		try:
			self.optimizer.load_state_dict(model_file['optimizer'])
		except:
			pass


	def load_data(self, filepath):
		""" Only load training masks """
		data_file = torch.load(os.path.join(filepath))  # , map_location=self.device)
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

