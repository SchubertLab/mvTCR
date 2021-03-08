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
from .cnn import CNNEncoder, CNNDecoder
from .bigru import BiGRUEncoder, BiGRUDecoder
from .transformer import TransformerEncoder, TransformerDecoder
from .mlp import MLP
from .losses.nb import NB
from .losses.kld import KLD
from .mlp_scRNA import build_mlp_encoder, build_mlp_decoder


class JointModelTorch(nn.Module):
	def __init__(self, xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm, seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams):
		super(JointModelTorch, self).__init__()

		seq_models = {'CNN': [CNNEncoder, CNNDecoder],
					  'Transformer': [TransformerEncoder, TransformerDecoder],
					  'BiGRU': [BiGRUEncoder, BiGRUDecoder]}

		scRNA_models = {'MLP': [build_mlp_encoder, build_mlp_decoder]}

		self.seq_encoder = seq_models[seq_model_arch][0](seq_model_hyperparams, hdim, num_seq_labels)
		self.seq_decoder = seq_models[seq_model_arch][1](seq_model_hyperparams, hdim*2, num_seq_labels)

		self.gene_encoder = scRNA_models[scRNA_model_arch][0](scRNA_model_hyperparams, xdim, hdim)
		self.gene_decoder = scRNA_models[scRNA_model_arch][1](scRNA_model_hyperparams, xdim, hdim)

		self.shared_encoder = MLP(hdim*2, zdim*2, shared_hidden, activation, activation, dropout, batch_norm, regularize_last_layer=False)  # zdim*2 because we predict mu and sigma simultaneously
		self.shared_decoder = MLP(zdim, hdim*2, shared_hidden[::-1], activation, activation, dropout, batch_norm, regularize_last_layer=True)

	def forward(self, scRNA, tcr_seq, tcr_len):
		"""
		Forward pass of autoencoder
		:param scRNA: torch.Tensor shape=[batch_size, num_genes]
		:param tcr_seq: torch.Tensor shape=[batch_size, seq_len, feature_dim]
		:return: scRNA_pred, tcr_seq_pred
		"""
		h_scRNA = self.gene_encoder(scRNA)  # shape=[batch_size, hdim]
		h_tcr_seq = self.seq_encoder(tcr_seq, tcr_len)  # shape=[batch_size, hdim]

		joint_feature = torch.cat([h_scRNA, h_tcr_seq], dim=-1)
		z_ = self.shared_encoder(joint_feature)  # shape=[batch_size, zdim*2]
		mu, logvar = z_[:, :z_.shape[1]//2], z_[:, z_.shape[1]//2:]  # mu.shape = logvar.shape = [batch_size, zdim]
		z = self.reparameterize(mu, logvar)  # shape=[batch_size, zdim]

		joint_dec_feature = self.shared_decoder(z)  # shape=[batch_size, hdim*2]
		scRNA_pred = self.gene_decoder(joint_dec_feature)  # shape=[batch_size, num_genes]
		tcr_seq_pred = self.seq_decoder(joint_dec_feature, tcr_seq)

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

		assert len(adatas) == len(names)
		assert len(adatas) == len(seq_keys) or len(seq_keys) == 0
		assert len(adatas) == len(gene_layers) or len(gene_layers) == 0

		self._train_history = defaultdict(list)
		self._val_history = defaultdict(list)
		self.seq_model_arch = seq_model_arch
		self.params = {'seq_model_arch': seq_model_arch, 'seq_model_hyperparams': seq_model_hyperparams,
					   'scRNA_model_arch': scRNA_model_arch, 'scRNA_model_hyperparams': scRNA_model_hyperparams,
					   'zdim': zdim, 'hdim': hdim, 'activation': activation, 'dropout': dropout, 'batch_norm': batch_norm,
					   'shared_hidden': shared_hidden}
		self.aa_to_id = aa_to_id

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
		num_seq_labels = len(aa_to_id)
		self.model = JointModelTorch(xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
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
			  validate_every=10,
			  print_every=10,
			  save_every=100,
			  num_workers=0,
			  verbose=1,
			  continue_training=False,
			  device=None,
			  comet=None
			  ):

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		print('Create Dataloader')
		# Initialize dataloader
		if continue_training:
			train_masks = self.load_data(f'../saved_models/{experiment_name}_data.pt')
			train_datasets, val_datasets, _ = self.create_datasets(self.adatas, self.names, self.gene_layers, self.seq_keys, val_split, metadata, train_masks)
		else:
			train_datasets, val_datasets, train_masks = self.create_datasets(self.adatas, self.names, self.gene_layers, self.seq_keys, val_split, metadata)
			self.save_data(f'../saved_models/{experiment_name}_data.pt', train_masks)

		train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, collate_fn=None, num_workers=num_workers)
		val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False, collate_fn=None, num_workers=num_workers)
		print('Dataloader created')

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
			TCR_criterion = nn.CrossEntropyLoss(ignore_index=self.aa_to_id['_'])
		else:
			raise ValueError(f'{losses[1]} loss in not implemented')

		KL_criterion = KLD()

		self.model = self.model.to(device)
		self.model.train()

		self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

		self.best_loss = 99999999999
		self.epoch = 0

		if n_iters is not None:
			n_epochs = n_iters // len(train_dataloader)

		if continue_training:
			# Load model and optimizer state_dict, as well as epoch and history
			self.load(f'../saved_models/{experiment_name}_last_model.pt')
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

				scRNA_loss = loss_weights[0] * scRNA_criterion(scRNA_pred, scRNA)
				# Before feeding in the gt_seq, the start token needs to get removed.
				# Further batch and seq dimension needs to be flatten
				TCR_loss = loss_weights[1] * TCR_criterion(tcr_seq_pred.flatten(end_dim=1), tcr_seq[:, 1:].flatten())
				KLD_loss = loss_weights[2] * KL_criterion(mu, logvar)
				loss = scRNA_loss + TCR_loss + KLD_loss

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				loss_train_total.append(loss.detach())
				scRNA_loss_train_total.append(scRNA_loss.detach())
				TCR_loss_train_total.append(TCR_loss.detach())
				KLD_loss_train_total.append(KLD_loss.detach())

				# some more metric

			if e % print_every == 0:
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

						scRNA_loss = loss_weights[0] * scRNA_criterion(scRNA_pred, scRNA)
						TCR_loss = loss_weights[1] * TCR_criterion(tcr_seq_pred.flatten(end_dim=1), tcr_seq[:, 1:].flatten())
						KLD_loss = loss_weights[2] * KL_criterion(mu, logvar)
						loss = scRNA_loss + TCR_loss + KLD_loss

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
						self.save(f'../saved_models/{experiment_name}_best_model.pt')

			if e % print_every == 0:
				self.save(f'../saved_models/{experiment_name}_last_model.pt')

			if e % save_every == 0:
				self.save(f'../saved_models/{experiment_name}_epoch_{str(e).zfill(3)}.pt')

	def get_latent(self, adatas, names, batch_size, num_workers=0, gene_layers=[], seq_keys=[], metadata=[], device=None):
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

	def create_datasets(self, adatas, names, layers, seq_keys, val_split, metadata=[], train_masks=None):
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
		model_file = {'state_dict': self.model.state_dict(),
					'train_history': self._train_history,
					'val_history': self._val_history,
					'epoch': self.epoch + 1,
					'aa_to_id': self.aa_to_id,
					'params': self.params,
					'best_loss': self.best_loss}
		try:
			model_file['optimizer'] = self.optimizer.state_dict()
		except:
			pass
		torch.save(model_file, filepath)

	def load(self, filepath):
		model_file = torch.load(os.path.join(filepath))  # , map_location=self.device)
		self.model.load_state_dict(model_file['state_dict'])
		self._train_history = model_file['train_history']
		self._val_history = model_file['val_history']
		self.epoch = model_file['epoch']
		self.aa_to_id = model_file['aa_to_id']
		self.best_loss = model_file['best_loss']
		try:
			self.optimizer.load_state_dict(model_file['optimizer'])
		except:
			pass

	def save_data(self, filepath, train_masks):
		data = {#'raw_data': [self.adatas, self.names, self.gene_layers, self.seq_keys],
				'train_masks': train_masks}
		torch.save(data, filepath)

	def load_data(self, filepath):
		data_file = torch.load(os.path.join(filepath))  # , map_location=self.device)
		# self.adatas, self.names, self.gene_layers, self.seq_keys = data_file['raw_data']
		# it returns train masks
		return data_file['train_masks']