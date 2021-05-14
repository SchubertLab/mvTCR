import scanpy as sc
import os
import yaml


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


def create_model(data, params, model_type):
    """
    Create a VAE model.
    :param data: adata object containing the training data
    :param params: hyperparameter of the model
    :param model_type: Class indicating which type of model was used (e.g. MoE)
    :return: pytorch model
    """
    model = model_type(
        adatas=[data],  # adatas containing gene expression and TCR-seq
        names=['bcc'],
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
        seq_keys=[]  # [] or list of str for seq keys of each dataset
    )
    return model
