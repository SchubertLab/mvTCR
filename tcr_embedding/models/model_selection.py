import optuna
import os
import importlib

import tcr_embedding.utils_training as utils


def get_parameter_functions(model_name, optimization_mode):
    model_name = model_name.lower()

    if optimization_mode not in ['pseudo_metric', 'knn_prediction', 'modulation_prediction']:
        model_name += '_equal'
    config_module = importlib.import_module(model_name)
    init_params = getattr(config_module, 'init_params')
    suggest_params = getattr(config_module, 'suggest_params')
    return init_params, suggest_params


def get_direction(optimization_mode):
    modes = {
        'pseudo_metric': 'maximize',
        'knn_prediction': 'maximize',
        'modulation_prediction': 'maximize',
        'reconstruction': 'minimize',
    }
    if optimization_mode in modes:
        return modes[optimization_mode]
    # reconstruction as default
    return 'minimize'


def complete_params_experiment(params):
    default_values = {
        'device': None,
        'balanced_sampling': None,
        'metadata': None,
        'conditional': None,
        'label_key': None,
        'n_epochs': 100,
        'kl_annealing_epochs': None,
        'early_stop': None,
        'save_path': '../saved_models/'
    }
    for key, value in default_values:
        if key not in params:
            params[key] = value
    return params


def objective(trial, adata, suggest_params, params_experiment, optimization_mode_params):
    params_experiment = complete_params_experiment(params_experiment)

    params_architecture = suggest_params(trial)
    if 'rna_weight' in optimization_mode_params:
        rna_weight = optimization_mode_params['rna_weight']
        params_architecture['loss_weights'] = params_architecture['loss_weights'].append(rna_weight)

    comet = utils.initialize_comet(params_architecture, params_experiment)

    model = utils.select_model_by_name(params_experiment['model_name'])
    model = model(adata, params_architecture, params_experiment['balanced_sampling'], params_experiment['metadata'],
                  params_experiment['conditional'], optimization_mode_params,
                  params_experiment['label_key'], params_experiment['device'])

    model.train(params_experiment['n_epochs'], params_architecture['batch_size'], params_architecture['learning_rate'],
                params_architecture['loss_weights'], params_experiment['kl_annealing_epochs'],
                params_experiment['early_stop'], params_experiment['save_path'], comet)

    # plot UMAPs
    model_names = ['reconstruction']
    if optimization_mode_params['name'] != 'reconstruction':
        model_names.append('metric')
    for state in model_names:
        model.load(os.path.join(params_experiment['save_path'], f'best_model_by_{state}.pt'))
        for subset in ['train', 'val']:
            adata_tmp = adata[adata.obs['set'] == subset]
            latent = model.get_latent(adata_tmp, params_experiment['metadata'], True)
            title = f''
            figs = utils.plot_umap_list(latent, title, params_experiment['metadata'])
            for fig, group in zip(figs, params_experiment['metadata']):
                comet.log_figure(f'{title}_{group}', fig)
    comet.end()
    return model.best_optimization_metric


def run_model_selection(adata, params_experiment, params_optimization, study_name, num_samples, timeout=None):
    sampler = optuna.samplers.TPESampler(seed=42)  # Make the sampler behave in a deterministic way.

    direction = get_direction(params_optimization['name'])

    study = optuna.create_study(study_name=study_name, sampler=sampler, storage=params_experiment['save_path'],
                                direction=direction, load_if_exists=False)

    init_params, suggest_params = get_parameter_functions(params_experiment['model_name'], params_optimization['name'])
    study.enqueue_trial(init_params)
    study.optimize(lambda trial: objective(trial, adata, suggest_params, params_experiment, params_optimization),
                   n_trials=num_samples, timeout=timeout)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print('Study statistics:')
    print(f'  Number of finished trials: {len(study.trials)}')
    print(f'  Number of pruned trials: {len(pruned_trials)}')
    print(f'  Number of complete trials: {len(complete_trials)}')

    best_trial = study.best_trial
    print('Best trial: ')
    print(f'  trial_{best_trial.number}')
    print(f'  Value: {best_trial.value}')
