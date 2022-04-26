# A Variational Information Bottleneck Approach to Multi-Omics Data Integration
import torch
import torch.nn as nn
import scanpy as sc

from tcr_embedding.models.architectures.transformer import TransformerEncoder, TransformerDecoder
from tcr_embedding.models.architectures.mlp import MLP
from tcr_embedding.models.architectures.mlp_scRNA import build_mlp_encoder, build_mlp_decoder
from tcr_embedding.models.vae_base_model import VAEBaseModel
from tcr_embedding.dataloader.DataLoader import initialize_prediction_loader


class MoEModelTorch(nn.Module):
	def __init__(self, tcr_params, rna_params, joint_params):
		super(MoEModelTorch, self).__init__()
		self.beta_only = 'beta_only' in tcr_params and tcr_params['beta_only']
		self.amount_chains = 1 if self.beta_only else 2

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
		use_embedding_for_cond = joint_params['use_embedding_for_cond'] if 'use_embedding_for_cond' in joint_params else True

		num_seq_labels = tcr_params['num_seq_labels']

		if not self.beta_only:
			self.alpha_encoder = TransformerEncoder(tcr_params, hdim // 2, num_seq_labels)
			self.alpha_decoder = TransformerDecoder(tcr_params, hdim, num_seq_labels)

		self.beta_encoder = TransformerEncoder(tcr_params, hdim // self.amount_chains, num_seq_labels)
		self.beta_decoder = TransformerDecoder(tcr_params, hdim, num_seq_labels)

		self.rna_encoder = build_mlp_encoder(rna_params, xdim, hdim)
		self.rna_decoder = build_mlp_decoder(rna_params, xdim, hdim)

		if cond_dim > 0:
			if use_embedding_for_cond:
				self.cond_emb = torch.nn.Embedding(num_conditional_labels, cond_dim)
			else:  # use one hot encoding
				self.cond_emb = None
				cond_dim = num_conditional_labels
		self.cond_input = cond_input
		self.use_embedding_for_cond = use_embedding_for_cond
		self.num_conditional_labels = num_conditional_labels
		cond_input_dim = cond_dim if cond_input else 0

		self.tcr_vae_encoder = MLP(hdim + cond_input_dim, zdim * 2, shared_hidden, activation, 'linear', dropout,
								   batch_norm, regularize_last_layer=False)
		self.tcr_vae_decoder = MLP(zdim + cond_dim, hdim, shared_hidden[::-1], activation, activation, dropout,
								   batch_norm, regularize_last_layer=True)

		self.rna_vae_encoder = MLP(hdim + cond_input_dim, zdim * 2, shared_hidden, activation, 'linear', dropout,
								   batch_norm, regularize_last_layer=False)
		self.rna_vae_decoder = MLP(zdim + cond_dim, hdim, shared_hidden[::-1], activation, activation, dropout,
								   batch_norm, regularize_last_layer=True)

		# used for NB loss
		self.theta = torch.nn.Parameter(torch.randn(xdim))

	def forward(self, rna, tcr, tcr_len, conditional=None):
		"""
		Forward pass of autoencoder
		:param rna: torch.Tensor shape=[batch_size, num_genes]
		:param tcr: torch.Tensor shape=[batch_size, seq_len, feature_dim]
		:param tcr_len: torch.Tensor shape=[batch_size]
		:param conditional: torch.Tensor shape=[batch_size, n_cond] one-hot-encoded conditional covariates
		:return:
			z: list of sampled latent variable zs. z = [z_rna, z_tcr, z_joint]
			mu: list of predicted means mu. mu = [mu_rna, mu_tcr, mu_joint]
			logvar: list of predicted logvars. logvar = [logvar_rna, logvar_tcr, logvar_joint]
			rna_pred: list of reconstructed rna. rna_pred = [rna_pred using z_rna, rna_pred using z_joint]
			tcr_pred: list of reconstructed tcr. tcr_pred = [tcr_pred using z_tcr, tcr_pred using z_joint]
		"""
		if conditional is not None:
			cond_emb_vec = self.cond_emb(conditional) if self.use_embedding_for_cond else torch.nn.functional.one_hot(conditional, self.num_conditional_labels)
		# Encode TCR

		beta_seq = tcr[:, tcr.shape[1] // 2 * (self.amount_chains == 2):]
		beta_len = tcr_len[:, self.amount_chains - 1]
		h_beta = self.beta_encoder(beta_seq, beta_len)  # shape=[batch_size, hdim//2]

		if not self.beta_only:
			alpha_seq = tcr[:, :tcr.shape[1] // 2]
			alpha_len = tcr_len[:, 0]
			h_alpha = self.alpha_encoder(alpha_seq, alpha_len)  # shape=[batch_size, hdim//2]
			h_tcr = torch.cat([h_alpha, h_beta], dim=-1)  # shape=[batch_size, hdim]
		else:
			h_tcr = torch.cat([h_beta], dim=-1)

		if conditional is not None and self.cond_input:
			h_tcr = torch.cat([h_tcr, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]

		# Encode RNA
		h_rna = self.rna_encoder(rna)  # shape=[batch_size, hdim]
		if conditional is not None and self.cond_input:
			h_rna = torch.cat([h_rna, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]

		# Predict Latent space
		z_rna_ = self.rna_vae_encoder(h_rna)  # shape=[batch_size, zdim*2]
		mu_rna, logvar_rna = z_rna_[:, :z_rna_.shape[1] // 2], z_rna_[:, z_rna_.shape[1] // 2:]
		z_rna = self.reparameterize(mu_rna, logvar_rna)  # shape=[batch_size, zdim]

		z_tcr_ = self.tcr_vae_encoder(h_tcr)  # shape=[batch_size, zdim*2]
		mu_tcr, logvar_tcr = z_tcr_[:, :z_tcr_.shape[1] // 2], z_tcr_[:, z_tcr_.shape[1] // 2:]
		z_tcr = self.reparameterize(mu_tcr, logvar_tcr)  # shape=[batch_size, zdim]

		z = [z_rna, z_tcr]
		mu = [mu_rna, mu_tcr]
		logvar = [logvar_rna, logvar_tcr]

		# Reconstruction
		rna_pred = []
		for z_ in z:
			if conditional is not None:
				z_ = torch.cat([z_, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]
			f_rna = self.rna_vae_decoder(z_)
			rna_pred.append(self.rna_decoder(f_rna))
		tcr_pred = []
		for z_ in z:
			if conditional is not None:
				z_ = torch.cat([z_, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]
			f_tcr = self.tcr_vae_decoder(z_)
			beta_pred = self.beta_decoder(f_tcr, beta_seq)

			if not self.beta_only:
				alpha_pred = self.alpha_decoder(f_tcr, alpha_seq)
				tcr_pred.append(torch.cat([alpha_pred, beta_pred], dim=1))
			else:
				tcr_pred.append(torch.cat([beta_pred], dim=1))
		return z, mu, logvar, rna_pred, tcr_pred

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
		transcriptome_pred = self.rna_vae_decoder(z_shared)
		transcriptome_pred = self.rna_decoder(transcriptome_pred)
		return transcriptome_pred

	def get_latent_from_z(self, z):
		z = 0.5 * (z[0] + z[1])
		return z


class MoEModel(VAEBaseModel):
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
		super(MoEModel, self).__init__(adata, params_architecture, balanced_sampling, metadata,
									   conditional, optimization_mode_params, label_key, device)
		self.model_type = 'moe'

		self.params_tcr['max_tcr_length'] = adata.obsm['alpha_seq'].shape[1]
		self.params_tcr['num_seq_labels'] = len(self.aa_to_id)

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

		self.model = MoEModelTorch(self.params_tcr, self.params_rna, self.params_joint)

	def calculate_loss(self, rna_pred, rna, tcr_pred, tcr):
		rna_loss = self.loss_function_rna(rna_pred[0], rna) + self.loss_function_rna(rna_pred[1], rna)
		rna_loss *= 0.5 * self.loss_weights[0]

		# For GRU and Transformer, as they don't predict start token for alpha and beta chain, so amount of chains used
		if tcr_pred[0].shape[1] == tcr.shape[1] - self.model.amount_chains:
			mask = torch.ones_like(tcr).bool()
			mask[:, [0]] = False
			if not self.beta_only:
				mask[:, [mask.shape[1] // 2]] = False
			tcr_loss = self.loss_function_tcr(tcr_pred[0].flatten(end_dim=1), tcr[mask].flatten())
			if not 'beta_only' in self.params_tcr or self.params_tcr['beta_only']:
				tcr_loss += self.loss_function_tcr(tcr_pred[1].flatten(end_dim=1), tcr[mask].flatten())
			tcr_loss *= 0.5 * self.loss_weights[1]
		else:  # For CNN, as it predicts start token
			tcr_loss = (self.loss_function_tcr(tcr_pred[0].flatten(end_dim=1), tcr.flatten()) +
						self.loss_function_tcr(tcr_pred[1].flatten(end_dim=1), tcr.flatten()))
			tcr_loss *= 0.5 * self.loss_weights[1]
		return rna_loss, tcr_loss

	def calculate_kld_loss(self, mu, logvar, epoch):
		kld_loss = (self.loss_function_kld(mu[0], logvar[0]) + self.loss_function_kld(mu[1], logvar[1]))
		kld_loss *= 0.5 * self.loss_weights[2] * self.get_kl_annealing_factor(epoch)
		z = 0.5 * (mu[0] + mu[1])
		return kld_loss, z

	def get_latent_unimodal(self, adata, metadata, modality, return_mean=True):
		"""
		Get unimodal latent either from RNA or TCR only
		:param adata:
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param return_mean: bool, calculate latent space without sampling
		:return: adata containing embedding vector in adata.X for each cell and the specified metadata in adata.obs
		"""
		data_embed = initialize_prediction_loader(adata, metadata, self.batch_size, beta_only=self.beta_only)

		zs = []
		with torch.no_grad():
			self.model = self.model.to(self.device)
			self.model.eval()
			for rna, tcr, seq_len, _, labels, conditional in data_embed:
				rna = rna.to(self.device)
				tcr = tcr.to(self.device)

				if self.conditional is not None:
					conditional = conditional.to(self.device)
				else:
					conditional = None
				z, mu, _, _, _ = self.model(rna, tcr, seq_len, conditional)
				if return_mean:
					z = mu
				if modality == 'RNA':
					z = z[0]
				else:
					z = z[1]
				z = sc.AnnData(z.detach().cpu().numpy())
				# z.obs[metadata] = np.array(metadata_batch).T
				zs.append(z)
		latent = sc.AnnData.concatenate(*zs)
		latent.obs.index = adata.obs.index
		latent.obs[metadata] = adata.obs[metadata]
		return latent
