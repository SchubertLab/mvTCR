import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import scanpy as sc

from tcr_embedding.datasets.scdataset import TCRDataset
from .cnn import CNNEncoder, CNNDecoder
from .bigru import BiGRUEncoder, BiGRUDecoder
from .transformer import TransformerEncoder, TransformerDecoder
from .mlp import MLP
from .losses.nb import NB
from .losses.kld import KLD


class JointModelTorch(nn.Module):
	def __init__(self, xdim, hdim, zdim, gene_hidden, shared_hidden, activation, output_activation, dropout, batch_norm, seq_model_arch, seq_model_hyperparams):
		super(JointModelTorch, self).__init__()

		if seq_model_arch == 'CNN':
			self.seq_encoder = CNNEncoder(seq_model_hyperparams, hdim)
			self.seq_decoder = CNNDecoder(seq_model_hyperparams, hdim)
		elif seq_model_arch == 'Transformer':
			self.seq_encoder = TransformerEncoder(seq_model_hyperparams, hdim)
			self.seq_decoder = TransformerDecoder(seq_model_hyperparams, hdim)
		elif seq_model_arch == 'BiGRU':
			self.seq_encoder = BiGRUEncoder(seq_model_hyperparams, hdim)
			self.seq_decoder = BiGRUDecoder(seq_model_hyperparams, hdim)
		elif seq_model_arch == 'dummy':  # TODO dummy model, used before we have sequence representation, takes in scRNA data instead
			self.seq_encoder = MLP(xdim, hdim, gene_hidden, activation, activation, dropout, batch_norm, regularize_last_layer=True)
			self.seq_decoder = MLP(hdim*2, xdim, gene_hidden[::-1], activation, activation, dropout, batch_norm, regularize_last_layer=False)
		else:
			raise ValueError('Sequence architecture currently not supported, please try one of the following ["CNN", "Transformer", "BiGRU"]')

		self.gene_encoder = MLP(xdim, hdim, gene_hidden, activation, activation, dropout, batch_norm, regularize_last_layer=True)
		self.shared_encoder = MLP(hdim*2, zdim*2, shared_hidden, activation, activation, dropout, batch_norm, regularize_last_layer=False)  # zdim*2 because we predict mu and sigma simultaneously

		self.shared_decoder = MLP(zdim, hdim*2, shared_hidden[::-1], activation, activation, dropout, batch_norm, regularize_last_layer=True)
		self.gene_decoder = MLP(hdim*2, xdim, gene_hidden[::-1], activation, output_activation, dropout, batch_norm, regularize_last_layer=False)

	def forward(self, scRNA, tcr_seq):
		"""
		Forward pass of autoencoder
		:param scRNA: torch.Tensor shape=[batch_size, num_genes]
		:param tcr_seq: torch.Tensor shape=[batch_size, seq_len, feature_dim]
		:return: scRNA_pred, tcr_seq_pred
		"""
		h_scRNA = self.gene_encoder(scRNA)  # shape=[batch_size, hdim]
		h_tcr_seq = self.seq_encoder(tcr_seq)  # shape=[batch_size, hdim]

		joint_feature = torch.cat([h_scRNA, h_tcr_seq], dim=-1)
		z_ = self.shared_encoder(joint_feature)  # shape=[batch_size, zdim*2]
		mu, logvar = z_[:, :z_.shape[1]//2], z_[:, z_.shape[1]//2:]  # mu.shape = logvar.shape = [batch_size, zdim]
		z = self.reparameterize(mu, logvar)  # shape=[batch_size, zdim]

		joint_dec_feature = self.shared_decoder(z)  # shape=[batch_size, hdim*2]
		scRNA_pred = self.gene_decoder(joint_dec_feature)  # shape=[batch_size, num_genes]
		tcr_seq_pred = self.seq_decoder(joint_dec_feature)

		return z, mu, logvar, scRNA_pred, tcr_seq_pred

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


class JointModel():
	def __init__(self,
				 adatas,  # adatas containing gene expression and TCR-seq
				 names,
				 seq_model_arch,  # seq model architecture
				 seq_model_hyperparams,  # dict of seq model hyperparameters
				 zdim,  # zdim
				 hdim,  # hidden dimension of encoder for each modality
				 activation,
				 output_activation,
				 dropout,
				 batch_norm,
				 gene_hidden=[],  # hidden layers of gene encoder/ decoder
				 shared_hidden=[],
				 gene_layers=[],
				 seq_keys=[],
				 device=None  # torch.device
				 ):

		assert len(adatas) == len(names)
		assert len(adatas) == len(seq_keys) or len(seq_keys) == 0
		assert len(adatas) == len(gene_layers) or len(gene_layers) == 0

		self._train_history = defaultdict(list)
		self._val_history = defaultdict(list)
		self.seq_model_arch = seq_model_arch

		if device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = device

		self.adatas = adatas
		self.names = names

		if len(seq_keys) == 0:
			self.seq_keys = ['tcr_seq'] * len(adatas)
		else:
			self.seq_keys = seq_keys

		if len(gene_layers) == 0:
			self.gene_layers = [None] * len(adatas)
		else:
			self.gene_layers = gene_layers

		# assuming now gene expressions between datasets are using the same genes, so input vectors have same length and order, e.g. using inner or outer join
		xdim = adatas[0].X.shape[1] if self.gene_layers[0] is None else len(adatas[0].layers[self.gene_layers[0]].shape[1])

		self.model = JointModelTorch(xdim, hdim, zdim, gene_hidden, shared_hidden, activation, output_activation,
									 dropout, batch_norm, seq_model_arch, seq_model_hyperparams)

	def train(self,
			  experiment_name='no_name',
			  n_iters=None,
			  n_epochs=100,
			  batch_size=64,
			  lr=3e-4,
			  losses=['MSE', 'CE'],
			  loss_weights=[],
			  val_split=0.1,
			  validate_every=10,
			  print_every=10,
			  num_workers=0,
			  verbose=1,
			  ):

		# Initialize Data
		train_datasets, val_datasets = self.create_datasets(self.adatas, self.names, self.gene_layers, self.seq_keys, val_split)
		train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, collate_fn=None, num_workers=num_workers)
		val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False, collate_fn=None, num_workers=num_workers)

		try:
			os.makedirs('../saved_models/')  # Create directory to prevent Error while saving model weights
		except:
			pass

		# Initialize Loss and Optimizer
		assert len(losses) == 2, 'losses needs to contain two elements corresponding to scRNA and TCR-seq loss'

		if len(loss_weights) == 0:
			loss_weights = [1.0] * 3
		elif len(loss_weights) == 3:
			loss_weights = loss_weights
		else:
			raise ValueError(f'length of loss_weights must be 3 or [].')

		if losses[0] == 'MSE':
			scRNA_criterion = nn.MSELoss()
		elif losses[0] == 'NB':
			scRNA_criterion = NB()
		else:
			raise ValueError(f'{losses[0]} loss in not implemented')

		if losses[1] == 'CE':
			TCR_criterion = nn.CrossEntropyLoss()
		else:
			raise ValueError(f'{losses[1]} loss in not implemented')

		KL_criterion = KLD()

		optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

		self.model = self.model.to(self.device)
		self.model.train()

		if n_iters is not None:
			n_epochs = n_iters // len(train_dataloader)

		best_loss = 99999999999
		for e in tqdm(range(n_epochs), 'Epoch: '):

			# TRAIN LOOP
			loss_total = 0
			scRNA_loss_total = 0
			TCR_loss_total = 0
			KLD_loss_total = 0
			for scRNA, tcr_seq, name, index in train_dataloader:
				scRNA = scRNA.to(self.device)
				# tcr_seq = tcr_seq.to(self.device)  TODO uncomment after TCR sequence representation works

				if self.seq_model_arch == 'dummy':  # dummy model takes in scRNA instead of tcr_seq, just for debugging
					z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, scRNA)

					scRNA_loss = scRNA_criterion(scRNA, scRNA_pred)
					TCR_loss = scRNA_criterion(scRNA, tcr_seq_pred)  # Be careful! This is only for dummy model and usually doesn't make sense
					KLD_loss = KL_criterion(mu, logvar)
					loss = loss_weights[0] * scRNA_loss + loss_weights[1] * TCR_loss + loss_weights[2] * KLD_loss

				else:   # TODO uncomment after TCR sequence representation works
					z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, tcr_seq)

					scRNA_loss = scRNA_criterion(scRNA, scRNA_pred)
					TCR_loss = TCR_criterion(tcr_seq, tcr_seq_pred)
					KLD_loss = KL_criterion(mu, logvar)
					loss = loss_weights[0] * scRNA_loss + loss_weights[1] * TCR_loss + loss_weights[2] * KLD_loss

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				loss_total += loss.detach()
				scRNA_loss_total += scRNA_loss.detach()
				TCR_loss_total += TCR_loss.detach()
				KLD_loss_total += KLD_loss.detach()

				# some more metric

			if e % print_every == 0:
				self._train_history['epoch'].append(e)
				self._train_history['loss'].append(loss.detach().item())
				self._train_history['scRNA_loss'].append(scRNA_loss_total.item())
				self._train_history['TCR_loss'].append(TCR_loss_total.item())
				self._train_history['KLD_loss'].append(KLD_loss_total.item())

				if verbose >= 2:
					print(f'Train Loss: {loss_total / len(train_dataloader)}')
					print(f'Train scRNA Loss: {scRNA_loss_total / len(train_dataloader)}')
					print(f'Train TCR Loss: {TCR_loss_total / len(train_dataloader)}')
					print(f'Train KLD Loss: {KLD_loss_total / len(train_dataloader)}\n')

			# VALIDATION LOOP
			if e % validate_every == 0:
				with torch.no_grad():
					self.model.eval()
					val_loss = 0
					scRNA_loss_total = 0
					TCR_loss_total = 0
					KLD_loss_total = 0

					for scRNA, tcr_seq, name, index in val_dataloader:
						scRNA = scRNA.to(self.device)
						# tcr_seq = tcr_seq.to(self.device)  TODO uncomment after TCR sequence representation works

						if self.seq_model_arch == 'dummy':
							z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, scRNA)

							scRNA_loss = scRNA_criterion(scRNA, scRNA_pred)
							TCR_loss = scRNA_criterion(scRNA, tcr_seq_pred)  # Be careful! This is only for dummy model and usually doesn't make sense
							KLD_loss = KL_criterion(mu, logvar)
							loss = loss_weights[0] * scRNA_loss + loss_weights[1] * TCR_loss + loss_weights[2] * KLD_loss

						else:  # TODO uncomment after TCR sequence representation works
							z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, tcr_seq)

							scRNA_loss = scRNA_criterion(scRNA, scRNA_pred)
							TCR_loss = TCR_criterion(tcr_seq, tcr_seq_pred)
							KLD_loss = KL_criterion(mu, logvar)
							loss = loss_weights[0] * scRNA_loss + loss_weights[1] * TCR_loss + loss_weights[2] * KLD_loss

						val_loss += loss
						scRNA_loss_total += scRNA_loss
						TCR_loss_total += TCR_loss
						KLD_loss_total += KLD_loss

					self.model.train()

					self._val_history['epoch'].append(e)
					self._val_history['loss'].append(val_loss.item())
					self._val_history['scRNA_loss'].append(scRNA_loss_total.item())
					self._val_history['TCR_loss'].append(TCR_loss_total.item())
					self._val_history['KLD_loss'].append(KLD_loss_total.item())
					if verbose >= 1:
						print(f'Val Loss: {val_loss / len(val_dataloader)}')
						print(f'Val scRNA Loss: {scRNA_loss_total / len(val_dataloader)}')
						print(f'Val TCR Loss: {TCR_loss_total / len(val_dataloader)}')
						print(f'Val KLD Loss: {KLD_loss_total / len(val_dataloader)}')

					if val_loss < best_loss:
						best_loss = val_loss
						torch.save(self.model.state_dict(), f'../saved_models/{experiment_name}_best_model.pt')

	def predict(self, adatas, names, batch_size, num_workers=0, gene_layers=[], seq_keys=[]):
		if len(seq_keys) == 0:
			seq_keys = ['tcr_seq'] * len(adatas)
		else:
			seq_keys = seq_keys

		if len(gene_layers) == 0:
			gene_layers = [None] * len(adatas)
		else:
			gene_layers = gene_layers

		pred_datasets, _ = self.create_datasets(adatas, names, gene_layers, seq_keys, val_split=0)
		pred_dataloader = DataLoader(pred_datasets, batch_size=batch_size, shuffle=False, collate_fn=None, num_workers=num_workers)

		zs = []
		with torch.no_grad():
			self.model = self.model.to(self.device)
			self.model.eval()
			for scRNA, tcr_seq, name, index in tqdm(pred_dataloader, 'Batch: '):
				scRNA = scRNA.to(self.device)
				# tcr_seq = tcr_seq.to(self.device)  TODO uncomment after TCR sequence representation works

				if self.seq_model_arch == 'dummy':
					z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, scRNA)
				else:  # TODO uncomment after TCR sequence representation works
					z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, tcr_seq)
				z = sc.AnnData(z.detach().cpu().numpy())
				z.obs['barcode'] = index
				z.obs['dataset'] = name
				zs.append(z)

		return sc.AnnData.concatenate(*zs)

	def create_datasets(self, adatas, names, layers, seq_keys, val_split):
		dataset_names_train = []
		dataset_names_val = []
		scRNA_datas_train = []
		scRNA_datas_val = []
		seq_datas_train = []
		seq_datas_val = []
		adatas_train = {}
		adatas_val = {}
		index_train = []
		index_val = []

		# Iterates through datasets with corresponding dataset name, scRNA layer and TCR column key
		# Splits everything into train and val
		for i, (name, adata, layer, seq_key) in enumerate(zip(names, adatas, layers, seq_keys)):
			# Create train mask for each dataset separately
			num_samples = adata.X.shape[0] if layer is None else len(adata.layers[layer].shape[0])
			train_mask = np.zeros(num_samples, dtype=np.bool)
			train_size = int(num_samples * (1 - val_split))
			if val_split != 0:
				train_mask[:train_size] = 1
			else:
				train_mask[:] = 1
			np.random.shuffle(train_mask)

			# Save dataset splits
			scRNA_datas_train.append(adata.X[train_mask] if layer is None else adata.layers[layer][train_mask])
			scRNA_datas_val.append(adata.X[~train_mask] if layer is None else adata.layers[layer][~train_mask])

			seq_datas_train.append(adata.obs[seq_key][train_mask])
			seq_datas_val.append(adata.obs[seq_key][~train_mask])

			adatas_train[name] = adata[train_mask]
			adatas_val[name] = adata[~train_mask]

			dataset_names_train += [name] * adata[train_mask].shape[0]
			dataset_names_val += [name] * adata[~train_mask].shape[0]

			index_train += adata[train_mask].obs.index.to_list()
			index_val += adata[~train_mask].obs.index.to_list()

		train_dataset = TCRDataset(scRNA_datas_train, seq_datas_train, adatas_train, dataset_names_train, index_train)
		val_dataset = TCRDataset(scRNA_datas_val, seq_datas_val, adatas_val, dataset_names_val, index_val)

		return train_dataset, val_dataset

	@property
	def history(self):
		return pd.DataFrame(self._val_history)

	def save(self, filepath):
		torch.save({'weights': self.model.state_dict()}, filepath)

	def load(self, filepath):
		model_file = torch.load(os.path.join(filepath), map_location=self.device)
		self.model.load_state_dict(model_file['state_dict'])
