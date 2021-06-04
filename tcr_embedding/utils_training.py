import comet_ml
from comet_ml import Experiment
import warnings
import scanpy as sc
import os
import yaml

from datetime import datetime

# import ray
# from ray import tune

import tcr_embedding.models as models


def load_data(source='10x'):
    """
    Loads the whole dataset from a defined source.
    :param source: str indicting the dataset or filename starting from the data_folder
    :return: adata object
    """

    path_current = os.path.dirname(__file__)
    path_base = os.path.join(path_current, '../data/')

    if source == '10x':
        path_source = '10x_CD8TC/v6_supervised.h5ad'
    elif source == 'bcc':
        path_source = 'BCC/06_bcc_highly_var_5000.h5ad'
    elif source == 'covid':
        raise NotImplementedError
    else:
        path_source = source
    path_file = os.path.join(path_base, path_source)

    try:
        data = sc.read_h5ad(path_file)
    except FileNotFoundError:
        raise FileNotFoundError(f'Data file not found at {path_file}. '
                                f'Please specify data source by "10x", "bcc" or "covid". '
                                f'Alternatively, as filename starting from the data folder.')
    return data


def load_model(data, path_config, path_model, model_type):
    path_base = os.path.dirname(__file__)
    path_config = os.path.join(path_base, '..', path_config)
    path_model = os.path.join(path_base, '..', path_model)
    params = load_params(path_config)
    model = create_model(data, params, model_type)
    model.load(path_model)
    return model


def load_params(path_config):
    """
    Load previous configuration of the model
    :param path_config: path to the config yaml file
    :return: dictionary containing the hyperparameter of the model
    """
    path_config = path_config + 'params.json'
    with open(path_config) as config_file:
        config = yaml.load(config_file)
    config = correct_params(config)
    return config


def correct_params(params):
    """Ray Tune can't sample within lists, so this helper function puts the values back into lists
    :param params: hyperparameter dict
    :return: corrected hyperparameter dict
    """
    params['loss_weights'] = [params['loss_weights_scRNA'], params['loss_weights_seq'], params['loss_weights_kl']]
    params['shared_hidden'] = [params['shared_hidden']]
    if 'num_layers' in params:
        params['shared_hidden'] = params['shared_hidden'] * params['num_layers']

    if 'loss_scRNA' in params:
        params['losses'][0] = params['loss_scRNA']

    if 'gene_hidden' in params['scRNA_model_hyperparams']:
        params['scRNA_model_hyperparams']['gene_hidden'] = [params['scRNA_model_hyperparams']['gene_hidden']]

    if 'num_layers' in params['scRNA_model_hyperparams']:
        params['scRNA_model_hyperparams']['gene_hidden'] = params['scRNA_model_hyperparams']['gene_hidden'] * \
                                                           params['scRNA_model_hyperparams']['num_layers']

    if params['seq_model_arch'] == 'CNN':
        params['seq_model_hyperparams']['num_features'] = [
            params['seq_model_hyperparams']['num_features_1'],
            params['seq_model_hyperparams']['num_features_2'],
            params['seq_model_hyperparams']['num_features_3']
        ]
        # Encoder
        params['seq_model_hyperparams']['encoder']['kernel'] = [
            params['seq_model_hyperparams']['encoder']['kernel_1'],
            params['seq_model_hyperparams']['encoder']['kernel_23'],
            params['seq_model_hyperparams']['encoder']['kernel_23']
        ]
        params['seq_model_hyperparams']['encoder']['stride'] = [
            params['seq_model_hyperparams']['encoder']['stride_1'],
            params['seq_model_hyperparams']['encoder']['stride_23'],
            params['seq_model_hyperparams']['encoder']['stride_23']
        ]
        # Decoder
        params['seq_model_hyperparams']['decoder']['kernel'] = [
            params['seq_model_hyperparams']['decoder']['kernel_1'],
            params['seq_model_hyperparams']['decoder']['kernel_2'],
        ]
        params['seq_model_hyperparams']['decoder']['stride'] = [
            params['seq_model_hyperparams']['decoder']['stride_1'],
            params['seq_model_hyperparams']['decoder']['stride_2'],
        ]
    return params


def create_model(data, params, model_type, name='bcc', params_additional=None):
    """
    Create a VAE model.
    :param data: adata object containing the training data
    :param params: hyperparameter of the model
    :param model_type: Class indicating which type of model was used (e.g. MoE)
    :param name: Name of the dataset
    :param params_additional: parameters needed to be passed eg for evaluation
    :return: pytorch model
    """
    model = model_type(
        adatas=[data],  # adatas containing gene expression and TCR-seq
        names=[name],
        aa_to_id=data.uns['aa_to_id'],  # dict {aa_char: id}
        seq_model_arch=params['seq_model_arch'],  # seq model architecture
        seq_model_hyperparams=params['seq_model_hyperparams'],  # dict of seq model hyperparameters
        scRNA_model_arch=params['scRNA_model_arch'],
        scRNA_model_hyperparams=params['scRNA_model_hyperparams'],
        zdim=params['zdim'],  # zdim
        hdim=params['hdim'],  # hidden dimension of scRNA and seq encoders
        activation=params['activation'],  # activation function of autoencoder hidden layers
        dropout=params['dropout'],
        batch_norm=params['batch_norm'],
        shared_hidden=params['shared_hidden'],  # hidden layers of shared encoder / decoder
        gene_layers=[],  # [] or list of str for layer keys of each dataset
        seq_keys=[],  # [] or list of str for seq keys of each dataset
        params_additional=params_additional
    )
    return model


def train_call(model, params_hpo, params_fixed, comet):
    n_epochs = params_fixed['n_epochs'] * params_hpo['batch_size'] // 256
    early_stop = params_fixed['early_stop'] * params_hpo['batch_size'] // 256
    epoch2step = 256 / params_hpo['batch_size']
    epoch2step *= 1000
    save_every = n_epochs // params_fixed['num_checkpoints']
    if save_every == 0:
        save_every = 1
    model.train(
        experiment_name=params_fixed['name'],
        n_iters=None,
        n_epochs=n_epochs,
        batch_size=params_hpo['batch_size'],
        lr=params_hpo['lr'],
        losses=params_hpo['losses'],
        loss_weights=params_hpo['loss_weights'],
        kl_annealing_epochs=None,
        val_split='set',
        metadata=params_fixed['metadata'],
        early_stop=early_stop,
        balanced_sampling=params_fixed['balanced_sampling'],
        validate_every=params_fixed['validate_every'],
        save_every=save_every,
        save_path=params_fixed['save_path'],
        save_last_model=False,
        num_workers=0,
        device=None,
        comet=comet,
        tune=tune
    )


def initialize_comet(params_hpo, params_fixed):
    if params_fixed['comet'] is None:
        return None

    current_datetime = datetime.now().strftime("%Y%m%d-%H.%M")
    experiment_name = params_fixed['name']  # + '_' + current_datetime

    path_key = os.path.join(os.path.dirname(__file__), '../comet_ml_key/API_key.txt')
    with open(path_key) as f:
        comet_key = f.read()
    experiment = Experiment(api_key=comet_key, workspace=params_fixed['workspace'], project_name=experiment_name)

    experiment.log_parameters(params_hpo)
    experiment.log_parameters(params_hpo['scRNA_model_hyperparams'], prefix='scRNA')
    experiment.log_parameters(params_hpo['seq_model_hyperparams'], prefix='seq')

    experiment.log_parameters(params_fixed, prefix='fixed_params')

    if params_hpo['seq_model_arch'] == 'CNN':
        experiment.log_parameters(params_hpo['seq_model_hyperparams']['encoder'], prefix='seq_encoder')
        experiment.log_parameters(params_hpo['seq_model_hyperparams']['decoder'], prefix='seq_decoder')
    return experiment


def select_model_by_name(model_name):
    """
    Select between modeltypes (e.g. single, concat, poe, moe, ...) by an identifier
    :param model_name:  str indicating the model type, type is chosen, when following indicator are part of the string
                        default: joint model
                        'seperate': for individual alpha beta chain representation
                        'moe': mixture of experts
                        'poe': product of experts
    :return: class of the corresponding model
    """
    if 'single' in model_name and 'separate' not in model_name:
        init_model = models.single_model.SingleModel
    elif 'moe' in model_name:
        init_model = models.moe.MoEModel
    elif 'poe' in model_name:
        init_model = models.poe.PoEModel
    elif 'separate' in model_name:
        init_model = models.separate_model.SeparateModel
    else:
        init_model = models.joint_model.JointModel
    return init_model


def init_model(params, model_type, adata, dataset_name):
    if model_type == 'RNA':
        init_model_func = models.single_model.SingleModel
    elif model_type == 'PoE':
        init_model_func = models.poe.PoEModel
    elif model_type == 'concat' or model_type == 'TCR':
        init_model_func = models.separate_model.SeparateModel
    else:
        raise NotImplementedError(f'The specified model {model_type} is not implemented, please try one of the follow ["RNA", "TCR", "concat", "PoE"]')

    if model_type == 'RNA' and params['seq_model_arch'] != 'None':
        warnings.warn('You specified RNA as model_type, but params contains TCR-seq hyperparameters, these will be ignored')
        params['seq_model_arch'] = 'None'
        params['seq_model_hyperparams'] = {}

    if model_type == 'TCR' and params['scRNA_model_arch'] != 'None':
        warnings.warn('You specified TCR as model_type, but params contains RNA-seq hyperparameters, these will be ignored')
        params['scRNA_model_arch'] = 'None'
        params['scRNA_model_hyperparams'] = {}

    model = init_model_func(
        adatas=[adata],  # adatas containing gene expression and TCR-seq
        names=[dataset_name],
        aa_to_id=adata.uns['aa_to_id'],  # dict {aa_char: id}
        seq_model_arch=params['seq_model_arch'],  # seq model architecture
        seq_model_hyperparams=params['seq_model_hyperparams'],  # dict of seq model hyperparameters
        scRNA_model_arch=params['scRNA_model_arch'],
        scRNA_model_hyperparams=params['scRNA_model_hyperparams'],
        zdim=params['zdim'],  # zdim
        hdim=params['hdim'],  # hidden dimension of scRNA and seq encoders
        activation=params['activation'],  # activation function of autoencoder hidden layers
        dropout=params['dropout'],
        batch_norm=params['batch_norm'],
        shared_hidden=params['shared_hidden'],  # hidden layers of shared encoder / decoder
        gene_layers=[],  # [] or list of str for layer keys of each dataset
        seq_keys=[],  # [] or list of str for seq keys of each dataset
    )

    return model


def show_umap(adata, test_embedding_func, color='binding_name', source_data='train', min_dist=0.5, spread=1.0, palette=None):

    adata = adata[adata.obs['set'] == source_data]
    latent = test_embedding_func(adata)
    adata.obsm['latent'] = latent
    sc.pp.neighbors(adata, use_rep='latent')
    sc.tl.umap(adata, min_dist=min_dist, spread=spread)
    sc.pl.umap(adata, color=color, palette=palette, ncols=1)


def determine_marker_genes(adata, resolution, visualize=False):
    adata = adata[adata.obs['set'] == 'train']
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.leiden(adata, resolution=resolution)
    if visualize:
        sc.tl.umap(adata)  # for visualization only
        sc.pl.umap(adata, color='leiden')

    # Filter TRA, TRB and TRG which forms the T-cell receptor
    adata = adata[:, ~((adata.var.index.str.contains('TRA')) | (adata.var.index.str.contains('TRB') | (adata.var.index.str.contains('TRG'))))]
    sc.tl.rank_genes_groups(adata, groupby='leiden')
    highly_variable = list(adata.uns['rank_genes_groups']['names'][0])

    return highly_variable

