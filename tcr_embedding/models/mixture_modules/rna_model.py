import torch
import torch.nn as nn

from tcr_embedding.models.architectures.mlp import MLP
from tcr_embedding.models.architectures.mlp_scRNA import build_mlp_encoder, build_mlp_decoder
from tcr_embedding.models.vae_base_model import VAEBaseModel


def none_model(hyperparams, hdim, xdim):
	pass


class RnaModelTorch(nn.Module):
	def __init__(self, rna_params, joint_params):
		super(RnaModelTorch, self).__init__()
		xdim = rna_params['xdim']
		hdim = joint_params['hdim']
		num_conditional_labels = joint_params['num_conditional_labels']
		cond_dim = joint_params['cond_dim']
		cond_input = joint_params['cond_input']
		zdim = joint_params['zdim']
		shared_hidden = joint_params['shared_hidden']
		activation = joint_params['activation']
		dropout = joint_params['dropout']
		batch_norm = joint_params['batch_norm']

		self.gene_encoder = build_mlp_encoder(rna_params, xdim, hdim)
		self.gene_decoder = build_mlp_decoder(rna_params, xdim, hdim)

		if num_conditional_labels > 0:
			self.cond_emb = torch.nn.Embedding(num_conditional_labels, cond_dim)
		self.cond_input = cond_input
		cond_input_dim = cond_dim if cond_input else 0

		self.shared_encoder = MLP(hdim + cond_input_dim, zdim * 2, shared_hidden, activation, 'linear',
								  dropout, batch_norm,   regularize_last_layer=False)
		self.shared_decoder = MLP(zdim + cond_dim, hdim, shared_hidden[::-1], activation, activation,
								  dropout, batch_norm, regularize_last_layer=True)

		# used for NB loss
		self.theta = torch.nn.Parameter(torch.randn(xdim))

	def forward(self, rna, tcr, tcr_len, conditional=None):
		"""
		Forward pass of autoencoder
		:param rna: torch.Tensor shape=[batch_size, num_genes]
		:param tcr: torch.Tensor shape=[batch_size, seq_len, num_seq_labels]
		:param tcr_len: torch.LongTensor shape=[batch_size] indicating how long the real unpadded length is
		:param conditional: torch.Tensor shape=[batch_size, n_cond] one-hot-encoded conditional covariates
		:return: scRNA_pred, tcr_seq_pred
		"""

		hidden_rna = self.gene_encoder(rna)

		if conditional is not None:
			cond_emb_vec = self.cond_emb(conditional)
			if self.cond_input:  # condition input flag is set
				hidden_rna = torch.cat([hidden_rna, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]

		z_ = self.shared_encoder(hidden_rna)  # shape=[batch_size, zdim*2]
		mu, logvar = z_[:, :z_.shape[1] // 2], z_[:, z_.shape[1] // 2:]  # mu.shape = logvar.shape = [batch_size, zdim]
		z = self.reparameterize(mu, logvar)  # shape=[batch_size, zdim]

		if conditional is not None:
			z_input = torch.cat([z, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+cond_emb_dim]
		else:
			z_input = z  # shape=[batch_size, hdim]
		joint_dec_feature = self.shared_decoder(z_input)  # shape=[batch_size, hdim]

		rna_pred = self.gene_decoder(joint_dec_feature)  # shape=[batch_size, num_genes]
		return z, mu, logvar, rna_pred, 0

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
		transcriptome_pred = self.gene_decoder(z_shared)
		return transcriptome_pred

	def get_latent_from_z(self, z):
		return z


class RnaModel(VAEBaseModel):
	def __init__(self,
				 adata,
				 params_architecture,
				 balanced_sampling='clonotype',
				 metadata=None,
				 conditional=None,
				 optimization_mode_params=None,
				 label_key=None,
				 device=None
				 ):

		super(RnaModel, self).__init__(adata, params_architecture, balanced_sampling, metadata,
									   conditional, optimization_mode_params, label_key, device)
		self.model_type = 'rna'

		self.params_rna['xdim'] = adata[0].X.shape[1]

		num_conditional_labels = 0
		cond_dim = 0
		if self.conditional is not None:
			if self.conditional in adata.obsm:
				num_conditional_labels = adata.obsm[self.conditional].shape[1]
			else:
				num_conditional_labels = len(adata.obs[self.conditional].unique())
			if 'c_embedding_dim' not in self.params_joint:
				cond_dim = 20
			else:
				cond_dim = self.params_joint['c_embedding_dim']
		self.params_joint['num_conditional_labels'] = num_conditional_labels
		self.params_joint['cond_dim'] = cond_dim
		self.params_joint['cond_input'] = conditional is not None

		self.model = RnaModelTorch(self.params_rna, self.params_joint)

	def calculate_loss(self, rna_pred, rna, tcr_pred, tcr):
		rna_loss = self.loss_weights[0] * self.loss_function_rna(rna_pred, rna)
		tcr_loss = torch.FloatTensor([0]).to(self.device)
		return rna_loss, tcr_loss

	def calculate_kld_loss(self, mu, logvar, epoch):
		kld_loss = self.loss_function_kld(mu, logvar)
		kld_loss *= self.loss_weights[2] * self.get_kl_annealing_factor(epoch)
		z = mu  # make z deterministic by using the mean
		return kld_loss, z
