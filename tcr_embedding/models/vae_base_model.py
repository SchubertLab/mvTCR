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

from tcr_embedding.datasets.scdataset import TCRDataset
from .losses.nb import NB
from .losses.kld import KLD


class VAEBaseModel:
	def __init__(self,
				 adatas,  # adatas containing gene expression and TCR-seq
				 names,
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
				 gene_layers=[],
				 seq_keys=[]
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
		"""

		assert len(adatas) == len(names)
		assert len(adatas) == len(seq_keys) or len(seq_keys) == 0
		assert len(adatas) == len(gene_layers) or len(gene_layers) == 0

		self.adatas = adatas
		self.names = names

		self._train_history = defaultdict(list)
		self._val_history = defaultdict(list)
		self.seq_model_arch = seq_model_arch

		if 'max_tcr_length' not in seq_model_hyperparams:
			seq_model_hyperparams['max_tcr_length'] = self.get_max_tcr_length()

		self.params = {'seq_model_arch': seq_model_arch, 'seq_model_hyperparams': seq_model_hyperparams,
					   'scRNA_model_arch': scRNA_model_arch, 'scRNA_model_hyperparams': scRNA_model_hyperparams,
					   'zdim': zdim, 'hdim': hdim, 'activation': activation, 'dropout': dropout, 'batch_norm': batch_norm,
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

		# assuming now gene expressions between datasets are using the same genes, so input vectors have same length and order, e.g. using inner or outer join
		xdim = adatas[0].X.shape[1] if self.gene_layers[0] is None else len(adatas[0].layers[self.gene_layers[0]].shape[1])
		num_seq_labels = len(aa_to_id)
		self.model = self.build_model(xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
									  seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams)

	def train(self,
			  experiment_name='example',
			  n_iters=None,
			  n_epochs=100,
			  batch_size=64,
			  lr=3e-4,
			  losses=['MSE', 'CE'],
			  loss_weights=[],
			  val_split='set',
			  metadata=[],
			  early_stop=None,
			  validate_every=10,
			  print_every=10,
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
		:param losses: list of str, losses[0] := Loss for scRNA reconstruction, losses[1] := Loss for TCR reconstruction
		:param loss_weights: list of floats, loss_weights[0]:=weight or scRNA loss, loss_weights[1]:=weight for TCR loss, loss_weights[2] := KLD Loss
		:param val_split: str or float, if str it indicates the adata.obs[val_split] containing 'train' and 'val', if float, adata is split randomly by val_split
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment #TODO maybe delete this
		:param early_stop: int, stop training after this number of epochs if val loss is not improving anymore
		:param validate_every: int, epochs to validate
		:param print_every: not used  # TODO maybe delete this, only here for backward compatibility
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
			train_datasets, val_datasets, _ = self.create_datasets(self.adatas, self.names, self.gene_layers, self.seq_keys, val_split, metadata, self.train_masks)
		else:
			train_datasets, val_datasets, self.train_masks = self.create_datasets(self.adatas, self.names, self.gene_layers, self.seq_keys, val_split, metadata)
			# self.save_data(f'../saved_models/{experiment_name}_data.pt', train_masks)
			# self.train_masks = train_masks

		train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, collate_fn=None, num_workers=num_workers)
		val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False, collate_fn=None, num_workers=num_workers)
		print('Dataloader created')

		try:
			os.makedirs(save_path)  # Create directory to prevent Error while saving model weights
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
			TCR_criterion = nn.CrossEntropyLoss(ignore_index=self.aa_to_id['_'])
		else:
			raise ValueError(f'{losses[1]} loss in not implemented')

		KL_criterion = KLD()

		self.model = self.model.to(device)
		self.model.train()

		self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

		self.best_loss = 99999999999
		no_improvements = 0
		self.epoch = 0

		if n_iters is not None:
			n_epochs = n_iters // len(train_dataloader)

		if continue_training:
			# Load model and optimizer state_dict, as well as epoch and history
			self.load(os.path.join(save_path, f'{experiment_name}_last_model.pt'))
			print(f'Continue training from epoch {self.epoch}')

		for e in tqdm(range(self.epoch, n_epochs), 'Epoch: '):
			self.epoch = e
			# TRAIN LOOP
			loss_train_total = []
			scRNA_loss_train_total = []
			TCR_loss_train_total = []
			KLD_loss_train_total = []
			for scRNA, tcr_seq, name, index, seq_len, metadata_batch in train_dataloader:
				scRNA = scRNA.to(device)
				tcr_seq = tcr_seq.to(device)

				z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, tcr_seq, seq_len)

				loss, scRNA_loss, TCR_loss, KLD_loss = self.calculate_loss(mu, logvar, scRNA_pred, scRNA, tcr_seq_pred, tcr_seq, loss_weights, scRNA_criterion, TCR_criterion, KL_criterion)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				loss_train_total.append(loss.detach())
				scRNA_loss_train_total.append(scRNA_loss.detach())
				TCR_loss_train_total.append(TCR_loss.detach())
				KLD_loss_train_total.append(KLD_loss.detach())

				# some more metric

			if e % validate_every == 0:
				loss_train_total = torch.stack(loss_train_total).mean().item()
				scRNA_loss_train_total = torch.stack(scRNA_loss_train_total).mean().item()
				TCR_loss_train_total = torch.stack(TCR_loss_train_total).mean().item()
				KLD_loss_train_total = torch.stack(KLD_loss_train_total).mean().item()

				self._train_history['epoch'].append(e)
				self._train_history['loss'].append(loss_train_total)
				self._train_history['scRNA_loss'].append(scRNA_loss_train_total)
				self._train_history['TCR_loss'].append(TCR_loss_train_total)
				self._train_history['KLD_loss'].append(KLD_loss_train_total)

				if verbose >= 2:
					print('\n')
					print(f'Train Loss: {loss_train_total}')
					print(f'Train scRNA Loss: {scRNA_loss_train_total}')
					print(f'Train TCR Loss: {TCR_loss_train_total}')
					print(f'Train KLD Loss: {KLD_loss_train_total}\n')
				if comet is not None:
					comet.log_metrics({'Train Loss': loss_train_total,
									   'Train scRNA Loss': scRNA_loss_train_total,
									   'Train TCR Loss': TCR_loss_train_total,
									   'Train KLD Loss': KLD_loss_train_total},
									  step=e, epoch=e)

			# VALIDATION LOOP
			if e % validate_every == 0:
				with torch.no_grad():
					self.model.eval()
					loss_val_total = []
					scRNA_loss_val_total = []
					TCR_loss_val_total = []
					KLD_loss_val_total = []

					for scRNA, tcr_seq, name, index, seq_len, metadata_batch in val_dataloader:
						scRNA = scRNA.to(device)
						tcr_seq = tcr_seq.to(device)

						z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, tcr_seq, seq_len)


						loss, scRNA_loss, TCR_loss, KLD_loss = self.calculate_loss(mu, logvar, scRNA_pred, scRNA, tcr_seq_pred, tcr_seq, loss_weights, scRNA_criterion, TCR_criterion, KL_criterion)

						loss_val_total.append(loss)
						scRNA_loss_val_total.append(scRNA_loss)
						TCR_loss_val_total.append(TCR_loss)
						KLD_loss_val_total.append(KLD_loss)

					self.model.train()

					loss_val_total = torch.stack(loss_val_total).mean().item()
					scRNA_loss_val_total = torch.stack(scRNA_loss_val_total).mean().item()
					TCR_loss_val_total = torch.stack(TCR_loss_val_total).mean().item()
					KLD_loss_val_total = torch.stack(KLD_loss_val_total).mean().item()

					self._val_history['epoch'].append(e)
					self._val_history['loss'].append(loss_val_total)
					self._val_history['scRNA_loss'].append(scRNA_loss_val_total)
					self._val_history['TCR_loss'].append(TCR_loss_val_total)
					self._val_history['KLD_loss'].append(KLD_loss_val_total)
					if verbose >= 1:
						print('\n')
						print(f'Val Loss: {loss_val_total}')
						print(f'Val scRNA Loss: {scRNA_loss_val_total}')
						print(f'Val TCR Loss: {TCR_loss_val_total}')
						print(f'Val KLD Loss: {KLD_loss_val_total}')

					if comet is not None:
						comet.log_metrics({'Val Loss': loss_val_total,
										   'Val scRNA Loss': scRNA_loss_val_total,
										   'Val TCR Loss': TCR_loss_val_total,
										   'Val KLD Loss': KLD_loss_val_total},
										  step=e, epoch=e)

					if loss_val_total < self.best_loss:
						self.best_loss = loss_val_total
						self.save(os.path.join(save_path, f'{experiment_name}_best_model.pt'))
						no_improvements = 0
					else:
						no_improvements += 1

			if e % validate_every == 0:
				self.save(os.path.join(save_path, f'{experiment_name}_last_model.pt'))

			if save_every is not None and e % save_every == 0:
				self.save(os.path.join(save_path, f'{experiment_name}_epoch_{str(e).zfill(3)}.pt'))

			if early_stop is not None and no_improvements > early_stop:
				break

	def get_latent(self, adatas, names, batch_size, num_workers=0, gene_layers=[], seq_keys=[], metadata=[], device=None):
		"""
		Get latent
		:param adatas: list of adatas
		:param names: list of str names for each adata, same order as adatas
		:param batch_size: int, batch size
		:param num_workers: int, num_workers for dataloader
		:param gene_layers: list of str or [], keys for scRNA data, i.e. adata.layer[gene_layers[i]] for each dataset i, or empty to use adata.X
		:param seq_keys: list of str or [], keys for TCR data, i.e. adata.obsm[seq_keys[i]] for each dataset i, or empty to use adata.obsm['tcr_seq']
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param device: None or str, if None device is determined automatically, based on if cuda.is_available
		:return: adata containing embedding vector in adata.X for each cell and the specified metadata in adata.obs
		"""
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		if len(seq_keys) == 0:
			seq_keys = ['tcr_seq'] * len(adatas)
		else:
			seq_keys = seq_keys

		if len(gene_layers) == 0:
			gene_layers = [None] * len(adatas)
		else:
			gene_layers = gene_layers

		pred_datasets, _, _ = self.create_datasets(adatas, names, gene_layers, seq_keys, val_split=0, metadata=metadata)
		pred_dataloader = DataLoader(pred_datasets, batch_size=batch_size, shuffle=False, collate_fn=None, num_workers=num_workers)

		zs = []
		with torch.no_grad():
			self.model = self.model.to(device)
			self.model.eval()
			for scRNA, tcr_seq, name, index, seq_len, metadata_batch in tqdm(pred_dataloader, 'Batch: '):
				scRNA = scRNA.to(device)
				tcr_seq = tcr_seq.to(device)

				z, mu, logvar, scRNA_pred, tcr_seq_pred = self.model(scRNA, tcr_seq, seq_len)
				z = sc.AnnData(z.detach().cpu().numpy())
				z.obs['barcode'] = index
				z.obs['dataset'] = name
				z.obs[metadata] = np.array(metadata_batch).T
				zs.append(z)

		return sc.AnnData.concatenate(*zs)

	def kNN(self, train_data, test_data, classes, n_neighbors, weights='distance'):
		"""
		Perform kNN using scikit-learn package
		:param train_data: annData with features in X and class labels in train_data.obs[classes]
		:param test_data: annData with features in X
		:param classes: key for class labels in train_data.obs[classes]
		:param n_neighbors: number of neighbors for kNN
		:param weights: kNN weights, either 'distance' or 'uniform'
		:return: Writes results into the column test_data.obs['pred_' + classes]
		"""
		clf = KNeighborsClassifier(n_neighbors, weights)
		X_train = train_data.X

		# Create categorical labels
		train_data.obs[classes + '_label'] = train_data.obs[classes].astype('category').cat.codes
		mapping = dict(enumerate(train_data.obs[classes].astype('category').cat.categories))
		y_train = train_data.obs[classes + '_label'].values

		clf.fit(X_train, y_train)

		test_data.obs['pred_labels'] = clf.predict(test_data.X)
		test_data.obs['pred_' + classes] = test_data.obs['pred_labels'].map(mapping)

	def antigen_imputation(self, train_adata, val_adata, class_label='binding_name', metadata=['binding_name', 'binding_label'], n_neighbors=5, weights='distance'):
		"""
		Perform antigen imputation using kNN
		:param train_adata: adata, at least has adata.obs[class_label] and usually adata.X is the embedding
		:param val_adata: adata, at least has adata.obs[class_label] and usually adata.X is the embedding
		:param class_label: str, column containing labels
		:param metadata: list of str, further metadata to be contained in result
		:param n_neighbors: int, number of neighbors to consider
		:param weights: 'distance' or 'uniform': see sklearn
		:return: val_adata with results from kNN in adata.obs['pred'+class_label]
		"""
		if not class_label in metadata:
			metadata.append(class_label)
		z_train = self.get_latent(
			adatas=[train_adata],
			names=['10x'],
			batch_size=256,
			num_workers=0,
			gene_layers=[],
			seq_keys=[],
			metadata=['binding_name', 'binding_label']
		)

		z_val = self.get_latent(
			adatas=[val_adata],
			names=['10x'],
			batch_size=256,
			num_workers=0,
			gene_layers=[],
			seq_keys=[],
			metadata=['binding_name', 'binding_label']
		)
		self.kNN(z_train, z_val, class_label, n_neighbors, weights)

		return z_val

	def create_datasets(self, adatas, names, layers, seq_keys, val_split, metadata=[], train_masks=None):
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

		train_dataset = TCRDataset(scRNA_datas_train, seq_datas_train, seq_len_train, adatas_train, dataset_names_train, index_train, metadata_train)
		val_dataset = TCRDataset(scRNA_datas_val, seq_datas_val, seq_len_val, adatas_val, dataset_names_val, index_val, metadata_val)

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
					  'epoch': self.epoch + 1,
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

	def build_model(self, xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
					seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams):
		raise NotImplementedError()

	def calculate_loss(self, mu, logvar, scRNA_pred, scRNA, tcr_seq_pred, tcr_seq, loss_weights, scRNA_criterion, TCR_criterion, KL_criterion):
		raise NotImplementedError