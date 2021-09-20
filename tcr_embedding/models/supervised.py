import torch
import torch.nn as nn
import numpy as np

from .joint_model import JointModelTorch
from .single_model import SingleModelTorch
from tcr_embedding.models.architectures.mlp import MLP
from .vae_base_model import VAEBaseModel
from tcr_embedding.datasets.scdataset import TCRDataset


class Classifier(nn.Module):
	def __init__(self, zdim, n_classes, params):
		super(Classifier, self).__init__()
		activation = params['activation']
		hidden = params['hidden']
		dropout = params['dropout']
		batch_norm = params['batch_norm']

		self.layers = MLP(zdim, n_classes, hidden, activation, dropout=dropout, output_activation='linear',
						  batch_norm=batch_norm, regularize_last_layer=False)
		# self.output_layer = nn.Softmax()  # not needed as we use nn.CrossEntropyLoss which includes Softmax already

	def forward(self, x):
		return self.layers(x)


class SupervisedModelTorch(nn.Module):
	def __init__(self, xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
				 seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams, classifier_params, n_labels):
		super(SupervisedModelTorch, self).__init__()

		self.scRNA_model_arch = scRNA_model_arch
		self.seq_model_arch = seq_model_arch

		if seq_model_arch == 'None' and scRNA_model_arch == 'None':
			raise RuntimeError('At least seq_model_arch or scRNA_model_arch needs to be a valid specification')
		elif (seq_model_arch != 'None' and scRNA_model_arch == 'None') or (seq_model_arch == 'None' and scRNA_model_arch != 'None'):
			self.VAE = SingleModelTorch(xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
									   seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams)
		elif seq_model_arch != 'None' and scRNA_model_arch != 'None':
			self.VAE = JointModelTorch(xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
									  seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams)
		else:  # this shouldn't happen, the above should be all possible cases
			raise RuntimeError('Unknown combination')

		self.classifier = Classifier(zdim, n_labels, params=classifier_params)
		# used for NB loss
		self.theta = torch.nn.Parameter(torch.randn(xdim))


	def forward(self, scRNA, tcr_seq, tcr_len):
		z, mu, logvar, scRNA_pred, tcr_seq_pred = self.VAE(scRNA, tcr_seq, tcr_len)

		return z, mu, logvar, scRNA_pred, tcr_seq_pred

	def classify(self, z):
		labels_pred = self.classifier(z)

		return labels_pred


class SupervisedModel(VAEBaseModel):
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
				 classifier_params=None,
				 label_key='high_count_binding_name'
				 ):

		super(SupervisedModel, self).__init__(adatas, aa_to_id, seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams,
											  zdim, hdim, activation, dropout, batch_norm, shared_hidden, names, gene_layers, seq_keys, classifier_params)

		xdim = adatas[0].X.shape[1] if self.gene_layers[0] is None else len(adatas[0].layers[self.gene_layers[0]].shape[1])
		num_seq_labels = len(aa_to_id)
		n_labels = len(adatas[0].obs[label_key].unique())
		self.label_key = label_key
		self.model_type = 'supervised'
		# annData saves the keys in dict as str, not int
		self.label_to_specificity = {int(k): v for k, v in adatas[0].uns['label_to_specificity'].items()}

		self.model = SupervisedModelTorch(xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
										  seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams,
										  classifier_params, n_labels)

	def create_datasets(self, adatas, names, layers, seq_keys, val_split, metadata=[], train_masks=None, label_key=None):
		"""
		Create torch Dataset, see above for the input
		:param adatas: list of adatas
		:param names: list of str
		:param layers:
		:param seq_keys:
		:param val_split:
		:param metadata:
		:param train_masks: None or list of train_masks: if None new train_masks are created, else the train_masks are used, useful for continuing training
		:param label_key: str, key to get the column with labels for semi-supervised training
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
		labels_train = []
		labels_val = []

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

			if label_key is not None:
				labels_train.append(adata.obs[label_key][train_mask].values)
				labels_val.append(adata.obs[label_key][~train_mask].values)
			else:
				labels_train = None
				labels_val = None

		train_dataset = TCRDataset(scRNA_datas_train, seq_datas_train, seq_len_train, adatas_train, dataset_names_train, index_train, metadata_train, labels_train)
		val_dataset = TCRDataset(scRNA_datas_val, seq_datas_val, seq_len_val, adatas_val, dataset_names_val, index_val, metadata_val, labels_val)

		return train_dataset, val_dataset, train_masks

	def calculate_loss(self, scRNA_pred, scRNA, tcr_seq_pred, tcr_seq, loss_weights, scRNA_criterion, TCR_criterion, size_factor):
		if self.model.scRNA_model_arch != 'None' and self.model.seq_model_arch == 'None':
			scRNA_loss = loss_weights[0] * self.calc_scRNA_rec_loss(scRNA_pred, scRNA, scRNA_criterion, size_factor, self.losses[0])
			loss = scRNA_loss
			TCR_loss = torch.FloatTensor([0])

		# Only TCR model
		# batch and seq dimension needs to be flatten
		elif self.model.seq_model_arch != 'None' and self.model.scRNA_model_arch == 'None':
			if tcr_seq_pred.shape[1] == tcr_seq.shape[1] - 1:  # For GRU and Transformer, as they don't predict start token
				TCR_loss = loss_weights[1] * TCR_criterion(tcr_seq_pred.flatten(end_dim=1), tcr_seq[:, 1:].flatten())
			else:  # For CNN, as it predicts start token
				TCR_loss = loss_weights[1] * TCR_criterion(tcr_seq_pred.flatten(end_dim=1), tcr_seq.flatten())

			loss = TCR_loss
			scRNA_loss = torch.FloatTensor([0])

		# Joint Model
		elif self.model.seq_model_arch != 'None' and self.model.scRNA_model_arch != 'None':
			scRNA_loss = loss_weights[0] * self.calc_scRNA_rec_loss(scRNA_pred, scRNA, scRNA_criterion, size_factor, self.losses[0])

			if tcr_seq_pred.shape[1] == tcr_seq.shape[1] - 1:  # For GRU and Transformer, as they don't predict start token
				TCR_loss = loss_weights[1] * TCR_criterion(tcr_seq_pred.flatten(end_dim=1), tcr_seq[:, 1:].flatten())
			else:  # For CNN, as it predicts start token
				TCR_loss = loss_weights[1] * TCR_criterion(tcr_seq_pred.flatten(end_dim=1), tcr_seq.flatten())
			loss = scRNA_loss + TCR_loss

		return loss, scRNA_loss, TCR_loss

