from comet_ml import Experiment

import scanpy as sc
import yaml
import tcr_embedding as tcr


PATH_BASE = '../'
PATH_SAVE = '../saved_models/'


def load_data():
    path_data = PATH_BASE + 'data/BCC/06_bcc_highly_var_5000.h5ad'
    data_full = sc.read_h5ad(path_data)
    data_full = data_full[data_full.obs['set'] != 'test']
    return data_full


def load_config(file_name):
    path_configs = PATH_BASE + 'config/' + file_name + '.yaml'
    with open(path_configs) as config_file:
        config = yaml.load(config_file)
    return config


def create_model(datas, params):
    model = tcr.models.joint_model.JointModel(
        adatas=[datas],  # adatas containing gene expression and TCR-seq
        names=['bcc'],
        aa_to_id=data_tc.uns['aa_to_id'],  # dict {aa_char: id}
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


def create_comet_experiment(params):
    experiment_name = 'test_01'
    with open('../comet_ml_key/API_key.txt') as f:
        COMET_ML_KEY = f.read()
    experiment = Experiment(api_key=COMET_ML_KEY, workspace='bcc', project_name='test')
    experiment.log_parameters(params)
    experiment.log_parameters(params['scRNA_model_hyperparams'], prefix='scRNA')
    experiment.log_parameters(params['seq_model_hyperparams'], prefix='seq')
    experiment.log_parameter('experiment_name', experiment_name)
    return experiment


def train_model(model, params, comet):
    model.train(
        experiment_name='test',
        n_iters=None,
        n_epochs=10,
        batch_size=params['batch_size'],
        lr=params['lr'],
        losses=params['losses'],  # list of losses for each modality: losses[0] := scRNA, losses[1] := TCR
        loss_weights=params['loss_weights'],  # [] or list of floats storing weighting of loss in order [scRNA, TCR, KLD]
        val_split='set',  # float or str, if float: split is determined automatically, if str: used as key for train-val column
        metadata=[],
        validate_every=1,
        save_every=1,
        num_workers=0,
        early_stop=20,
        save_path=PATH_SAVE,
        device=None,
        comet=comet
    )


data_tc = load_data()
params_test = load_config('transformer')
model_test = create_model(data_tc, params_test)
comet_experiment = create_comet_experiment(params_test)
train_model(model_test, params_test, comet_experiment)

figure_groups = ['patient', 'clonotype', 'cluster', 'cluster_tcr', 'treatment', 'response']
val_latent = model_test.get_latent([data_tc[data_tc.obs['set'] == 'val']], batch_size=512, metadata=figure_groups)
figures = tcr.utils.plot_umap_list(val_latent, title='test' + '_val_best_recon', color_groups=figure_groups)


"""
scp 03 felix.drost@sepp:/storage/groups/imm01/workspace/TCR_Embedding/data/BCC

"""