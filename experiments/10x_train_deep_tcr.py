from comet_ml import Experiment
import scanpy as sc
import yaml
import tcr_embedding as tcr
from tcr_embedding.evaluation.WrapperFunctions import get_model_prediction_function
from tcr_embedding.evaluation.Imputation import run_imputation_evaluation
from tqdm import tqdm
import os

PATH_BASE = '../'
PATH_SAVE = '../saved_models/'
EXPERIMENT_NAME = 'deep_tcr'

def load_data(filename):
    path_data = PATH_BASE + 'data/10x_CD8TC/' + filename
    data_full = sc.read_h5ad(path_data)
    data_full = data_full[data_full.obs['set'] != 'test']
    return data_full


def load_config(file_name):
    path_configs = PATH_BASE + 'config/' + file_name + '.yaml'
    with open(path_configs) as config_file:
        config = yaml.load(config_file)
    return config


def create_model(datas, params):
    model = tcr.models.deep_tcr.DeepTCR(
        adatas=[datas],  # adatas containing gene expression and TCR-seq
        names=['10x'],
        seq_model_hyperparams=params['seq_model_hyperparams'],  # dict of seq model hyperparameters
        zdim=params['zdim'],  # zdim
    )
    return model


def train_model(model, experiment):
    model.train(
        experiment_name=EXPERIMENT_NAME,
        n_iters=None,
        n_epochs=100000,
        batch_size=params['batch_size'],
        lr=params['lr'],
        loss_weights=params['loss_weights'],
        val_split='set',
        metadata=[],
        early_stop=500,
        validate_every=5,
        save_every=1000,
        save_path=PATH_SAVE,
        num_workers=0,
        verbose=1,
        continue_training=False,
        device=None,
        comet=experiment
    )


def evaluate_model(model, adata, params, experiment):
    epoch2step = 256 / params['batch_size']  # normalization factor of epoch -> step, as one epoch with different batch_size results in different numbers of iterations
    epoch2step *= 1000  # to avoid decimal points, as we multiply with a float number

    checkpoint_fps = os.listdir(PATH_SAVE)
    checkpoint_fps = [checkpoint_fp for checkpoint_fp in checkpoint_fps if f'{EXPERIMENT_NAME}_epoch_' in checkpoint_fp]
    checkpoint_fps.sort()
    for fp in tqdm(checkpoint_fps, 'kNN for previous checkpoints: '):
        model.load(os.path.join(PATH_SAVE, fp))
        test_embedding_func = get_model_prediction_function(model, batch_size=params['batch_size'])
        try:
            summary = run_imputation_evaluation(adata, test_embedding_func, query_source='val', use_non_binder=True, use_reduced_binders=True)
        except:
            break

        metrics = summary['knn']
        for antigen, metric in metrics.items():
            if antigen != 'accuracy':
                experiment.log_metrics(metric, prefix=antigen, step=int(model.epoch * epoch2step), epoch=model.epoch)
            else:
                experiment.log_metric('accuracy', metric, step=int(model.epoch * epoch2step), epoch=model.epoch)

    print('kNN for best reconstruction loss model')
    # Evaluate Model (best model based on reconstruction loss)
    model.load(os.path.join(PATH_SAVE, f'{EXPERIMENT_NAME}_best_model.pt'))
    test_embedding_func = get_model_prediction_function(model, batch_size=params['batch_size'])
    try:
        summary = run_imputation_evaluation(adata, test_embedding_func, query_source='val', use_non_binder=True, use_reduced_binders=True)
    except:
        return

    metrics = summary['knn']
    for antigen, metric in metrics.items():
        if antigen != 'accuracy':
            experiment.log_metrics(metric, prefix='best_recon_' + antigen, step=int(model.epoch * epoch2step), epoch=model.epoch)
        else:
            experiment.log_metric('best_recon_accuracy', metric, step=int(model.epoch * epoch2step), epoch=model.epoch)


with open('../comet_ml_key/API_key.txt') as f:
    COMET_ML_KEY = f.read()

experiment = Experiment(api_key=COMET_ML_KEY, workspace='tcr', project_name='10x_'+EXPERIMENT_NAME)
adata = load_data('v6_deep_tcr.h5ad')
params = load_config(EXPERIMENT_NAME)

experiment.log_parameters(params)
experiment.log_parameters(params['seq_model_hyperparams'], prefix='seq_model')
experiment.log_parameters(params['seq_model_hyperparams']['encoder'], prefix='seq_encoder')
experiment.log_parameters(params['seq_model_hyperparams']['decoder'], prefix='seq_decoder')

model = create_model(adata, params)
train_model(model, experiment)
evaluate_model(model, adata, params, experiment)
print('Finished')
