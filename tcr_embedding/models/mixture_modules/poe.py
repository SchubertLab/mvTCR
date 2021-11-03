# A Variational Information Bottleneck Approach to Multi-Omics Data Integration
import torch
import torch.nn as nn

from tcr_embedding.models.architectures.transformer import TransformerEncoder, TransformerDecoder
from tcr_embedding.models.architectures.mlp import MLP
from tcr_embedding.models.architectures.mlp_scRNA import build_mlp_encoder, build_mlp_decoder
from tcr_embedding.models.vae_base_model import VAEBaseModel


class PoEModelTorch(nn.Module):
    def __init__(self, tcr_params, rna_params, joint_params):
        super(PoEModelTorch, self).__init__()
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

        num_seq_labels = tcr_params['num_seq_labels']

        self.alpha_encoder = TransformerEncoder(tcr_params, hdim // 2, num_seq_labels)
        self.alpha_decoder = TransformerDecoder(tcr_params, hdim, num_seq_labels)

        self.beta_encoder = TransformerEncoder(tcr_params, hdim // 2, num_seq_labels)
        self.beta_decoder = TransformerDecoder(tcr_params, hdim, num_seq_labels)

        self.rna_encoder = build_mlp_encoder(rna_params, xdim, hdim)
        self.rna_decoder = build_mlp_decoder(rna_params, xdim, hdim)

        if cond_dim > 0:
            self.cond_emb = torch.nn.Embedding(num_conditional_labels, cond_dim)
        self.cond_input = cond_input
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
		:param tcr_len: torch.LongTensor shape=[batch_size] indicating how long the real unpadded length is
		:param conditional: torch.Tensor shape=[batch_size, n_cond] one-hot-encoded conditional covariates
		:return:
			z: list of sampled latent variable zs. z = [z_rna, z_tcr, z_joint]
			mu: list of predicted means mu. mu = [mu_rna, mu_tcr, mu_joint]
			logvar: list of predicted logvars. logvar = [logvar_rna, logvar_tcr, logvar_joint]
			rna_pred: list of reconstructed rna. rna_pred = [rna_pred using z_rna, rna_pred using z_joint]
			tcr_pred: list of reconstructed tcr. tcr_pred = [tcr_pred using z_tcr, tcr_pred using z_joint]
		"""
        if conditional is not None:
            cond_emb_vec = self.cond_emb(conditional)
        # Encode TCR
        alpha_seq = tcr[:, :tcr.shape[1] // 2]
        alpha_len = tcr_len[:, 0]

        beta_seq = tcr[:, tcr.shape[1] // 2:]
        beta_len = tcr_len[:, 1]

        h_alpha = self.alpha_encoder(alpha_seq, alpha_len)  # shape=[batch_size, hdim//2]
        h_beta = self.beta_encoder(beta_seq, beta_len)  # shape=[batch_size, hdim//2]
        h_tcr = torch.cat([h_alpha, h_beta], dim=-1)  # shape=[batch_size, hdim]
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

        # Predict joint latent space using PoE
        mu_joint, logvar_joint = self.product_of_experts(mu_rna, mu_tcr, logvar_rna, logvar_tcr)
        z_joint = self.reparameterize(mu_joint, logvar_joint)

        z = [z_rna, z_tcr, z_joint]
        mu = [mu_rna, mu_tcr, mu_joint]
        logvar = [logvar_rna, logvar_tcr, logvar_joint]

        # Reconstruction
        rna_pred = []
        for z_ in [z_rna, z_joint]:
            if conditional is not None:
                z_ = torch.cat([z_, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]
            f_rna = self.rna_vae_decoder(z_)
            rna_pred.append(self.rna_decoder(f_rna))
        tcr_pred = []
        for z_ in [z_tcr, z_joint]:
            if conditional is not None:
                z_ = torch.cat([z_, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]
            f_tcr = self.tcr_vae_decoder(z_)
            alpha_pred = self.alpha_decoder(f_tcr, alpha_seq)
            beta_pred = self.beta_decoder(f_tcr, beta_seq)

            tcr_pred.append(torch.cat([alpha_pred, beta_pred], dim=1))

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

    def product_of_experts(self, mu_rna, mu_tcr, logvar_rna, logvar_tcr):
        # formula: var_joint = inv(inv(var_prior) + sum(inv(var_modalities)))
        logvar_joint = 1.0 / torch.exp(logvar_rna) + 1.0 / torch.exp(
            logvar_tcr) + 1.0  # sum up all inverse vars, logvars first needs to be converted to var, last 1.0 is coming from the prior
        logvar_joint = torch.log(1.0 / logvar_joint)  # inverse and convert to logvar

        # formula: mu_joint = (mu_prior*inv(var_prior) + sum(mu_modalities*inv(var_modalities))) * var_joint, where mu_prior = 0.0
        mu_joint = mu_rna * (1.0 / torch.exp(logvar_rna)) + mu_tcr * (1.0 / torch.exp(logvar_tcr))
        mu_joint = mu_joint * torch.exp(logvar_joint)

        return mu_joint, logvar_joint

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
        return z[2]


class PoEModel(VAEBaseModel):
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
        super(PoEModel, self).__init__(adata, params_architecture, balanced_sampling, metadata,
                                       conditional, optimization_mode_params, label_key, device)

        self.model_type = 'poe'

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

        self.model = PoEModelTorch(self.params_tcr, self.params_rna, self.params_joint)

    def calculate_loss(self, rna_pred, rna, tcr_pred, tcr):
        # Evaluate same-modality and joint reconstruction
        rna_loss = self.loss_function_rna(rna_pred[0], rna) + self.loss_function_rna(rna_pred[1], rna)
        rna_loss *= 0.5 * self.loss_weights[0]

        # For GRU and Transformer, as they don't predict start token for alpha and beta chain, so -2
        if tcr_pred[0].shape[1] == tcr.shape[1] - 2:
            mask = torch.ones_like(tcr).bool()
            mask[:, [0, mask.shape[1] // 2]] = False
            tcr_loss = (self.loss_function_tcr(tcr_pred[0].flatten(end_dim=1), tcr[mask].flatten()) +
                        self.loss_function_tcr(tcr_pred[1].flatten(end_dim=1), tcr[mask].flatten()))
            tcr_loss *= 0.5 * self.loss_weights[1]
        else:  # For CNN, as it predicts start token
            tcr_loss = (self.loss_function_tcr(tcr_pred[0].flatten(end_dim=1), tcr.flatten()) +
                        self.loss_function_tcr(tcr_pred[1].flatten(end_dim=1), tcr.flatten()))
            tcr_loss *= 0.5 * self.loss_weights[1]
        return rna_loss, tcr_loss

    def calculate_kld_loss(self, mu, logvar, epoch):
        """
		Calculate the kld loss for rna, tcr, and joint latent space (optional between rna and joint)
		:param mu:
		:param logvar:
		:param epoch:
		:return:
		"""
        kld_loss = (self.loss_function_kld(mu[0], logvar[0])
                    + self.loss_function_kld(mu[1], logvar[1])
                    + self.loss_function_kld(mu[2], logvar[2]))
        kld_loss *= 1.0 / 3.0 * self.loss_weights[2] * self.get_kl_annealing_factor(epoch)

        # possible constrain on joint space to resemble more the TCR space
        if len(self.loss_weights) == 4:
            kld_rna_joint = self.loss_function_kld(mu[0], logvar[0], mu[2], logvar[2])
            kld_rna_joint = self.loss_weights[3] * kld_rna_joint
            kld_loss += kld_rna_joint
        z = mu[2]  # use joint latent variable for further downstream tasks
        return kld_loss, z
