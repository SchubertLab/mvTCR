import torch
import torch.nn as nn

from .cnn import CNNEncoder, CNNDecoder
from .bigru import BiGRUEncoder, BiGRUDecoder
from .transformer import TransformerEncoder, TransformerDecoder
from .mlp import MLP
from .mlp_scRNA import build_mlp_encoder, build_mlp_decoder
from .vae_base_model import VAEBaseModel


class MMVAETorch(nn.Module):
	def __init__(self, xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm, seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams):
		super(MMVAETorch, self).__init__()

		seq_models = {'CNN': [CNNEncoder, CNNDecoder],
					  'Transformer': [TransformerEncoder, TransformerDecoder],
					  'BiGRU': [BiGRUEncoder, BiGRUDecoder]}

		scRNA_models = {'MLP': [build_mlp_encoder, build_mlp_decoder]}

		self.tcr_encoder = seq_models[seq_model_arch][0](seq_model_hyperparams, hdim, num_seq_labels)
		self.tcr_decoder = seq_models[seq_model_arch][1](seq_model_hyperparams, hdim, num_seq_labels)

		self.rna_encoder = scRNA_models[scRNA_model_arch][0](scRNA_model_hyperparams, xdim, hdim)
		self.rna_decoder = scRNA_models[scRNA_model_arch][1](scRNA_model_hyperparams, xdim, hdim)

		self.tcr_vae_encoder = MLP(hdim, zdim * 2, shared_hidden, activation, 'linear', dropout, batch_norm, regularize_last_layer=False)  # zdim*2 because we predict mu and sigma simultaneously
		self.tcr_vae_decoder = MLP(zdim, hdim, shared_hidden[::-1], activation, activation, dropout, batch_norm, regularize_last_layer=True)

		self.rna_vae_encoder = MLP(hdim, zdim * 2, shared_hidden, activation, 'linear', dropout, batch_norm, regularize_last_layer=False)  # zdim*2 because we predict mu and sigma simultaneously
		self.rna_vae_decoder = MLP(zdim, hdim, shared_hidden[::-1], activation, activation, dropout, batch_norm, regularize_last_layer=True)

		# used for NB loss
		self.theta = torch.nn.Parameter(torch.randn(xdim))

	def forward(self, scRNA, tcr_seq, tcr_len):
		"""
		Forward pass of autoencoder
		:param scRNA: torch.Tensor shape=[batch_size, num_genes]
		:param tcr_seq: torch.Tensor shape=[batch_size, seq_len, feature_dim]
		:return: scRNA_pred, tcr_seq_pred
		"""
		h_scRNA = self.rna_encoder(scRNA)  # shape=[batch_size, hdim]
		h_tcr = self.tcr_encoder(tcr_seq, tcr_len)  # shape=[batch_size, hdim]

		z_scRNA_ = self.rna_vae_encoder(h_scRNA)  # shape=[batch_size, zdim*2]
		mu_scRNA, logvar_scRNA = z_scRNA_[:, :z_scRNA_.shape[1]//2], z_scRNA_[:, z_scRNA_.shape[1]//2:]  # mu.shape = logvar.shape = [batch_size, zdim]
		z_scRNA = self.reparameterize(mu_scRNA, logvar_scRNA)  # shape=[batch_size, zdim]

		z_tcr_ = self.tcr_vae_encoder(h_tcr)  # shape=[batch_size, zdim*2]
		mu_tcr, logvar_tcr = z_tcr_[:, :z_tcr_.shape[1]//2], z_tcr_[:, z_tcr_.shape[1]//2:]  # mu.shape = logvar.shape = [batch_size, zdim]
		z_tcr = self.reparameterize(mu_tcr, logvar_tcr)  # shape=[batch_size, zdim]

		z = [z_scRNA, z_tcr]
		mu = [mu_scRNA, mu_tcr]
		logvar = [logvar_scRNA, logvar_tcr]

		scRNA_pred = []
		tcr_seq_pred = []
		for z_ in z:
			f_rna = self.rna_vae_decoder(z_)
			scRNA_pred.append(self.rna_decoder(f_rna))
			f_tcr = self.tcr_vae_decoder(z_)
			tcr_seq_pred.append(self.tcr_decoder(f_tcr, tcr_len))

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


class MMVAE(VAEBaseModel):
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
				 params_additional=None
				 ):
		super(MMVAE, self).__init__(adatas, aa_to_id, seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams,
									zdim, hdim, activation, dropout, batch_norm, shared_hidden, names, gene_layers, seq_keys, params_additional)
		self.mmvae = True
		xdim = adatas[0].X.shape[1] if self.gene_layers[0] is None else len(adatas[0].layers[self.gene_layers[0]].shape[1])
		num_seq_labels = len(aa_to_id)
		self.model = MMVAETorch(xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
								seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams)

	def calculate_loss(self, scRNA_pred, scRNA, tcr_seq_pred, tcr_seq, loss_weights, scRNA_criterion, TCR_criterion, size_factor):
		scRNA_loss = 0.5 * loss_weights[0] * \
					 (self.calc_scRNA_rec_loss(scRNA_pred[0], scRNA, scRNA_criterion, size_factor, self.losses[0]) +
					  self.calc_scRNA_rec_loss(scRNA_pred[1], scRNA, scRNA_criterion, size_factor, self.losses[0]))

		if tcr_seq_pred[0].shape[1] == tcr_seq.shape[1] - 1:  # For GRU and Transformer, as they don't predict start token
			TCR_loss = 0.5 * loss_weights[1] * \
					   (TCR_criterion(tcr_seq_pred[0].flatten(end_dim=1), tcr_seq[:, 1:].flatten()) +
						TCR_criterion(tcr_seq_pred[1].flatten(end_dim=1), tcr_seq[:, 1:].flatten()))
		else:  # For CNN, as it predicts start token
			TCR_loss = 0.5 * loss_weights[1] * \
					   (TCR_criterion(tcr_seq_pred[0].flatten(end_dim=1), tcr_seq.flatten()) +
						TCR_criterion(tcr_seq_pred[1].flatten(end_dim=1), tcr_seq.flatten()))

		loss = scRNA_loss + TCR_loss

		return loss, scRNA_loss, TCR_loss
