import optuna
import os
import sys
import importlib

import tcr_embedding.utils_training as utils


def get_parameter_functions(model_name, optimization_mode):
    model_name = model_name.lower()

    if optimization_mode not in ['pseudo_metric', 'knn_prediction', 'modulation_prediction']:
        model_name += '_equal'

    path_module = os.path.join(os.path.dirname(__file__), '..', 'config_optuna')
    sys.path.append(path_module)
    model_name = os.path.join(model_name)
    config_module = importlib.import_module(model_name)
    # init_params = getattr(config_module, 'init_params')
    suggest_params = getattr(config_module, 'suggest_params')
    # return init_params, suggest_params
    return suggest_params


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
        'metadata': [],
        'conditional': None,
        'label_key': None,
        'n_epochs': 200,
        'kl_annealing_epochs': None,
        'early_stop': 5,
        'save_path': '../saved_models/',
        'model_name': 'moe',
    }
    for key, value in default_values.items():
        if key not in params:
            params[key] = value
    return params


def fail_save(func):
    # Keeps the HPO running, even when a training run fails.
    def wrapper(trial, adata, suggest_params, params_experiment, optimization_mode_params):
        direction = get_direction(optimization_mode_params['name'])
        try:
            return func(trial, adata, suggest_params, params_experiment, optimization_mode_params)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
            if direction == 'maximize':
                return 0.
            else:
                return 5.
    return wrapper


#@fail_save
def objective(trial, adata_tmp, suggest_params, params_experiment_base, optimization_mode_params):
    adata = adata_tmp.copy()
    params_experiment = params_experiment_base.copy()
    params_experiment = complete_params_experiment(params_experiment)
    params_experiment['save_path'] = os.path.join(params_experiment['save_path'], f'trial_{trial.number}')

    params_architecture = suggest_params(trial)

    if 'beta_only' in params_experiment and params_experiment['beta_only']:
        params_architecture['tcr']['beta_only'] = True

    if 'rna_weight' in optimization_mode_params:
        rna_weight = optimization_mode_params['rna_weight']
        params_architecture['loss_weights'] = params_architecture['loss_weights'].append(rna_weight)

    if 'use_embedding_for_cond' in params_experiment:
        params_architecture['joint']['use_embedding_for_cond'] = params_experiment['use_embedding_for_cond']
    # if 'cond_input' in params_experiment:
    #     params_architecture['joint']['cond_input'] = params_experiment['cond_input']

    comet = utils.initialize_comet(params_architecture, params_experiment)

    model = utils.select_model_by_name(params_experiment['model_name'])
    model = model(adata, params_architecture, params_experiment['balanced_sampling'], params_experiment['metadata'],
                  params_experiment['conditional'], optimization_mode_params,
                  params_experiment['label_key'], params_experiment['device'])

    model.train(params_experiment['n_epochs'], params_architecture['batch_size'], params_architecture['learning_rate'],
                params_architecture['loss_weights'], params_experiment['kl_annealing_epochs'],
                params_experiment['early_stop'], params_experiment['save_path'], comet)

    # plot UMAPs
    if comet is not None:
        model_names = ['reconstruction']
        if optimization_mode_params['name'] != 'reconstruction':
            model_names.append('metric')
        for state in model_names:
            model.load(os.path.join(params_experiment['save_path'], f'best_model_by_{state}.pt'))
            for subset in ['train', 'val']:
                adata_tmp = adata[adata.obs['set'] == subset]
                latent = model.get_latent(adata_tmp, params_experiment['metadata'], True)
                title = f'{state}_{subset}'
                figs = utils.plot_umap_list(latent, title, params_experiment['metadata'])
                if comet is not None:
                    for fig, group in zip(figs, params_experiment['metadata']):
                        comet.log_figure(f'{title}_{group}', fig)
        comet.end()
    return model.best_optimization_metric


def run_model_selection(adata, params_experiment, params_optimization, num_samples, timeout=None, n_jobs=1):
    sampler = optuna.samplers.TPESampler(seed=42)  # Make the sampler behave in a deterministic way.

    direction = get_direction(params_optimization['name'])

    storage = f'sqlite:///{params_experiment["save_path"]}.db'
    if os.path.exists(params_experiment['save_path'] + '.db'):
        os.remove(params_experiment['save_path'] + '.db')
    os.makedirs(os.path.dirname(params_experiment['save_path']), exist_ok=True)

    study = optuna.create_study(study_name=params_experiment['study_name'], sampler=sampler, storage=storage,
                                direction=direction, load_if_exists=False)

    suggest_params = get_parameter_functions(params_experiment['model_name'], params_optimization['name'])
    # study.enqueue_trial(init_params)
    study.optimize(lambda trial: objective(trial, adata, suggest_params, params_experiment, params_optimization),
                   n_trials=num_samples, timeout=timeout, n_jobs=n_jobs)

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
