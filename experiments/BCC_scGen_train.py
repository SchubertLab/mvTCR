import comet_ml
from comet_ml import Experiment

import tcr_embedding.utils_training as helper
from tcr_embedding.models.scGen import PertubationPredictor
from tcr_embedding.models.joint_model import JointModel
import tcr_embedding.utils as utils

import yaml
import os


def training_loop(params_hpo, params_scgen, params_fixed, model_type, comet):
    """
    Run 1 training for the BCC dataset
    :param params_hpo: tunable hyperparameter for this training run
    :param params_scgen: hyperparamter for the scGen process
    :param params_fixed: fixed parameters over a hyperparameter search
    :param model_type: Type of VAE model as class
    :param comet: Comet experiment where to log the data to
    :return: logs performance to comet or returns it
    """

    data = helper.load_data('bcc')
    for column, criteria in params_scgen['hold_out']:
        data = data[data.obs[column] != criteria]

    model = helper.create_model(data, params_hpo, model_type, 'scgen', params_scgen)

    helper.train_call(model, params_hpo, params_fixed, comet)
    print('training')
    if os.path.exists(os.path.join(params_fixed['save_path'], f'{params_fixed["name"]}_best_rec_model.pt')):
        model.load(os.path.join(params_fixed['save_path'], f'{params_fixed["name"]}_best_rec_model.pt'))

    performance = run_evaluation(model, data, params_scgen)
    print('eval')
    if comet is None:
        return performance
    plot_umaps(model, data, params_fixed, comet, 'best_recon_')
    print('umaps done')
    report_performance(performance, comet)
    print('reported')
    if os.path.exists(os.path.join(params_fixed['save_path'], f'{params_fixed["name"]}_best_gen_model.pt')):
        model.load(os.path.join(params_fixed['save_path'], f'{params_fixed["name"]}_best_gen_model.pt'))
        plot_umaps(model, data, params_fixed, comet, 'best_gen_')
    print('umaps2')
    return performance


def run_evaluation(model, data, params_scgen):
    """
    Evaluate Pertubation prediction for a trained model
    :param model: pytorch VAE model
    :param data: adata object containing the data
    :param params_scgen: dict describing the pertubation settings
    :return: dictionary containing a summary of the results
    """  # todo
    evaluator = PertubationPredictor(model, data=data, verbosity=0)
    result = evaluator.evaluate_pertubation(pertubation=params_scgen['pertubation'], splitting_criteria={'set': 'val'},
                                            per_column=params_scgen['per_column'], indicator=params_scgen['indicator'])
    return result


def plot_umaps(model, data, params_fixed, comet, prefix):
    figure_groups = params_fixed['metadata']
    val_latent = model.get_latent([data], batch_size=512, metadata=figure_groups)
    figures = utils.plot_umap_list(val_latent, title=params_fixed["name"] + f'_val_{prefix}',
                                   color_groups=figure_groups)
    for title, fig in zip(figure_groups, figures):
        comet.log_figure(figure_name=prefix + params_fixed["name"] + f'_val_{title}', figure=fig, step=model.epoch)


def report_performance(summary, comet):
    """
    Logs the performance to comet.
    :param summary: dict containing the performance report
    :param comet: comet experiment indicating the logging place
    :return: logs directly to comet
    """
    comet.log_parameters(summary['all_genes'], prefix='all_genes')
    comet.log_parameters(summary['top_100_genes'], prefix='top_100_genes')
    try:
        for name, summary in summary['per_cluster'].items():
            comet.log_parameters(summary, prefix=f'all_genes_{name}')
        for name, summary in summary['per_cluster_top_100'].items():
            comet.log_parameters(summary, prefix=f'top_100_genes_{name}')
    except KeyError:
        pass


if __name__ == '__main__':
    sc_gen_params = {
        'pertubation': 'treatment',
        'indicator': 'pre',
        'hold_out': [('patient', 'su009'),
                     ('patient', 'su006')],
        'per_column': None
    }

    fixed_params = {
        'name': 'test',
        'balanced_sampling': 'clonotype',
        'save_path': os.path.join(os.path.dirname(__file__), '../saved_models/test'),

        'metadata': ['patient', 'clonotype', 'cluster', 'cluster_tcr', 'treatment', 'response'],

        'n_epochs': 5,
        'early_stop': 100,
        'validate_every': 5,
        'num_checkpoints': 20,

        'tune': None,
        'comet': True,
        'workspace': 'bcc',
        'project': 'test',
    }
    with open("../config/transformer.yaml", 'r') as stream:
        hpo_params = yaml.safe_load(stream)

    training_loop(hpo_params, sc_gen_params, fixed_params, JointModel, comet=None)
