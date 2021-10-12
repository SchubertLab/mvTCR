import torch
import torch.nn as nn

from tcr_embedding.models.architectures.cnn import CNNEncoder, CNNDecoder
from tcr_embedding.models.architectures.bigru import BiGRUEncoder, BiGRUDecoder
from tcr_embedding.models.architectures.transformer import TransformerEncoder, TransformerDecoder
from tcr_embedding.models.architectures.mlp import MLP
from tcr_embedding.models.architectures.mlp_scRNA import build_mlp_encoder, build_mlp_decoder
from tcr_embedding.models.vae_base_model import VAEBaseModel


def none_model(hyperparams, hdim, xdim):
	pass


class SingleModelTorch(nn.Module):
	def __init__(self, xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm, seq_model_arch,
				 seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams, num_conditional_labels, cond_dim,
				 cond_input=False):
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

		if num_conditional_labels > 0:
			self.cond_emb = torch.nn.Embedding(num_conditional_labels, cond_dim)
		self.cond_input = cond_input
		cond_input_dim = cond_dim if cond_input else 0

		self.shared_encoder = MLP(hdim*num_modalities+cond_input_dim, zdim*2, shared_hidden, activation, 'linear', dropout, batch_norm, regularize_last_layer=False)  # zdim*2 because we predict mu and sigma simultaneously
		self.shared_decoder = MLP(zdim+cond_dim, hdim*num_modalities, shared_hidden[::-1], activation, activation, dropout, batch_norm, regularize_last_layer=True)

		# used for NB loss
		self.theta = torch.nn.Parameter(torch.randn(xdim))

	def forward(self, scRNA, tcr_seq, tcr_len, conditional=None):
		"""
		Forward pass of autoencoder
		:param scRNA: torch.Tensor shape=[batch_size, num_genes]
		:param tcr_seq: torch.Tensor shape=[batch_size, seq_len, num_seq_labels]
		:param tcr_len: torch.LongTensor shape=[batch_size] indicating how long the real unpadded length is
		:param conditional: torch.Tensor shape=[batch_size, n_cond] one-hot-encoded conditional covariates
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

		if conditional is not None:
			cond_emb_vec = self.cond_emb(conditional)
			if self.cond_input:  # condition input flag is set
				joint_feature = torch.cat([joint_feature, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]

		z_ = self.shared_encoder(joint_feature)  # shape=[batch_size, zdim*2]
		mu, logvar = z_[:, :z_.shape[1]//2], z_[:, z_.shape[1]//2:]  # mu.shape = logvar.shape = [batch_size, zdim]
		z = self.reparameterize(mu, logvar)  # shape=[batch_size, zdim]

		if conditional is not None:
			z_input = torch.cat([z, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+cond_emb_dim]
		else:
			z_input = z  # shape=[batch_size, hdim]
		joint_dec_feature = self.shared_decoder(z_input)  # shape=[batch_size, hdim]

		if self.scRNA_model_arch != 'None' and self.seq_model_arch == 'None':
			scRNA_pred = self.gene_decoder(joint_dec_feature)  # shape=[batch_size, num_genes]
			tcr_seq_pred = 0
		elif self.seq_model_arch != 'None' and self.scRNA_model_arch == 'None':
			tcr_seq_pred = self.seq_decoder(joint_dec_feature, tcr_seq)
			scRNA_pred = 0
		else:
			raise AssertionError('Please use Joint Model')

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

	def predict_transcriptome(self, z_shared, conditional=None):
		"""
		Predict the transcriptome connected to an shared latent space
		:param z_shared: torch.tensor, shared latent representation
		:param conditional:
		:return: torch.tensor, transcriptome profile
		"""
		if conditional is not None:  # more efficient than doing two concatenations
			cond_emb_vec = self.cond_emb(conditional)
			z_shared = torch.cat([z_shared, cond_emb_vec], dim=-1)  # shape=[batch_size, zdim+cond_dim]
		z_shared = self.shared_decoder(z_shared)
		if self.scRNA_model_arch != 'None' and self.seq_model_arch == 'None':
			transcriptome_pred = self.gene_decoder(z_shared)
		else:
			raise ValueError('Trying to predict transcriptome with a model without rna')
		return transcriptome_pred


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
				 seq_keys=[],
				 params_additional=None,
				 conditional=None,
				 optimization_mode='Reconstruction',
				 optimization_mode_params=None
				 ):

		super(SingleModel, self).__init__(adatas, aa_to_id, seq_model_arch, seq_model_hyperparams, scRNA_model_arch,
										  scRNA_model_hyperparams, zdim, hdim, activation, dropout, batch_norm,
										  shared_hidden, names, gene_layers, seq_keys, params_additional, conditional,
										  optimization_mode=optimization_mode, optimization_mode_params=optimization_mode_params)

		xdim = adatas[0].X.shape[1] if self.gene_layers[0] is None else len(adatas[0].layers[self.gene_layers[0]].shape[1])
		num_seq_labels = len(aa_to_id)
		if self.conditional is not None:
			if self.conditional in adatas[0].obsm:
				num_conditional_labels = adatas[0].obsm[self.conditional].shape[1]
			else:
				num_conditional_labels = len(adatas[0].obs[self.conditional].unique())
			try:
				cond_dim = params_additional['c_embedding_dim']
			except:
				cond_dim = 20
		else:
			num_conditional_labels = 0
			cond_dim = 0

		self.model = SingleModelTorch(xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
									  seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams,
									  num_conditional_labels, cond_dim)

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
