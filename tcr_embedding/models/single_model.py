import torch
import torch.nn as nn

from .cnn import CNNEncoder, CNNDecoder
from .bigru import BiGRUEncoder, BiGRUDecoder
from .transformer import TransformerEncoder, TransformerDecoder
from .mlp import MLP
from .mlp_scRNA import build_mlp_encoder, build_mlp_decoder
from .vae_base_model import VAEBaseModel


def none_model(hyperparams, hdim, xdim):
	pass


class SingleModelTorch(nn.Module):
	def __init__(self, xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm, seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams):
		super(SingleModelTorch, self).__init__()

		assert scRNA_model_arch != 'None' or seq_model_arch != 'None', 'At least scRNA- or seq-model needs to be not None'
		assert not(scRNA_model_arch == 'None' and seq_model_arch == 'None'), 'Please use the joint model'

		seq_models = {'CNN': [CNNEncoder, CNNDecoder],
					  'Transformer': [TransformerEncoder, TransformerDecoder],
					  'BiGRU': [BiGRUEncoder, BiGRUDecoder],
					  'None': [none_model, none_model]}

		scRNA_models = {'MLP': [build_mlp_encoder, build_mlp_decoder],
						'None': [none_model, none_model]}

		num_modalities = int(scRNA_model_arch != 'None') + int(seq_model_arch != 'None')

		self.scRNA_model_arch = scRNA_model_arch
		self.seq_model_arch = seq_model_arch

		self.seq_encoder = seq_models[seq_model_arch][0](seq_model_hyperparams, hdim, num_seq_labels)
		self.seq_decoder = seq_models[seq_model_arch][1](seq_model_hyperparams, hdim*num_modalities, num_seq_labels)

		self.gene_encoder = scRNA_models[scRNA_model_arch][0](scRNA_model_hyperparams, xdim, hdim)
		self.gene_decoder = scRNA_models[scRNA_model_arch][1](scRNA_model_hyperparams, xdim, hdim*num_modalities)

		self.shared_encoder = MLP(hdim*num_modalities, zdim*2, shared_hidden, activation, 'linear', dropout, batch_norm, regularize_last_layer=False)  # zdim*2 because we predict mu and sigma simultaneously
		self.shared_decoder = MLP(zdim, hdim*num_modalities, shared_hidden[::-1], activation, activation, dropout, batch_norm, regularize_last_layer=True)

		# used for NB loss
		self.theta = torch.nn.Parameter(torch.randn(xdim))

	def forward(self, scRNA, tcr_seq, tcr_len):
		"""
		Forward pass of autoencoder
		:param scRNA: torch.Tensor shape=[batch_size, num_genes]
		:param tcr_seq: torch.Tensor shape=[batch_size, seq_len, feature_dim]
		:return: scRNA_pred, tcr_seq_pred
		"""

		if self.scRNA_model_arch != 'None' and self.seq_model_arch == 'None':
			h_scRNA = self.gene_encoder(scRNA)  # shape=[batch_size, hdim]
			joint_feature = h_scRNA
		elif self.seq_model_arch != 'None' and self.scRNA_model_arch == 'None':
			h_tcr_seq = self.seq_encoder(tcr_seq, tcr_len)  # shape=[batch_size, hdim]
			joint_feature = h_tcr_seq
		else:
			raise AssertionError('Please use Joint Model')

		z_ = self.shared_encoder(joint_feature)  # shape=[batch_size, zdim*2]
		mu, logvar = z_[:, :z_.shape[1]//2], z_[:, z_.shape[1]//2:]  # mu.shape = logvar.shape = [batch_size, zdim]
		z = self.reparameterize(mu, logvar)  # shape=[batch_size, zdim]

		joint_dec_feature = self.shared_decoder(z)  # shape=[batch_size, hdim]

		if self.scRNA_model_arch != 'None' and self.seq_model_arch == 'None':
			scRNA_pred = self.gene_decoder(joint_dec_feature)  # shape=[batch_size, num_genes]
			tcr_seq_pred = 0
		elif self.seq_model_arch != 'None' and self.scRNA_model_arch == 'None':
			tcr_seq_pred = self.seq_decoder(joint_dec_feature, tcr_seq)
			scRNA_pred = 0

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


class SingleModel(VAEBaseModel):
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
				 seq_keys=[]
				 ):

		super(SingleModel, self).__init__(adatas, aa_to_id, seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams,
										  zdim, hdim, activation, dropout, batch_norm, shared_hidden, names, gene_layers, seq_keys)

	def build_model(self, xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
					seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams):
		return SingleModelTorch(xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
								seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams)

	def calculate_loss(self, scRNA_pred, scRNA, tcr_seq_pred, tcr_seq, loss_weights, scRNA_criterion, TCR_criterion, size_factor):
		# Only scRNA model
		if self.model.scRNA_model_arch != 'None' and self.model.seq_model_arch == 'None':
			scRNA_loss = loss_weights[0] * self.calc_scRNA_rec_loss(scRNA_pred, scRNA, scRNA_criterion, size_factor, self.losses[0])
			loss = scRNA_loss
			TCR_loss = torch.FloatTensor([0])

		# Only TCR model
		# batch and seq dimension needs to be flatten
		if self.model.seq_model_arch != 'None' and self.model.scRNA_model_arch == 'None':
			if tcr_seq_pred.shape[1] == tcr_seq.shape[1] - 1:  # For GRU and Transformer, as they don't predict start token
				TCR_loss = loss_weights[1] * TCR_criterion(tcr_seq_pred.flatten(end_dim=1), tcr_seq[:, 1:].flatten())
			else:  # For CNN, as it predicts start token
				TCR_loss = loss_weights[1] * TCR_criterion(tcr_seq_pred.flatten(end_dim=1), tcr_seq.flatten())

			loss = TCR_loss
			scRNA_loss = torch.FloatTensor([0])
		return loss, scRNA_loss, TCR_loss
