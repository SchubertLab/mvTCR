from comet_ml import Experiment
import scanpy as sc
import os
import argparse
import importlib
import warnings
from datetime import datetime
import optuna

import sys
sys.path.append('../')
sys.path.append('../config_optuna')
import tcr_embedding as tcr
from tcr_embedding.evaluation.WrapperFunctions import get_model_prediction_function
from tcr_embedding.evaluation.Clustering import run_clustering_evaluation

import tcr_embedding.utils_training as utils_train

import torch

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def objective(trial):
	"""
	Objective function for Optuna
	:param trial: Optuna Trial Object
	"""
	params = suggest_params(trial)

	save_path = f'../optuna/{name}/trial_{trial.number}'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	torch.save(params, os.path.join(save_path, 'params.pkl'))
	params_fixed['save_path'] = save_path

	experiment = utils_train.initialize_comet(params, params_fixed)

	adata = utils_train.load_data('scc')
	mask_holdout_donor = adata.obs['patient'] != params_fixed['holdout_donor']
	adata = adata[mask_holdout_donor]

	model = utils_train.init_model(params, model_type=params_fixed['model'], adata=adata,
								   dataset_name='haniffa', conditional=params_fixed['conditional'],
								   optimization_mode='scGen', optimization_mode_params=optimization_params)

	utils_train.train_call(model, params, params_fixed, experiment)

	color_groups = ['treatment', 'response', 'patient', 'cluster_tcr', 'cluster', 'clonotype']
	if os.path.exists(os.path.join(save_path, f'{name}_best_rec_model.pt')):
		# UMAP
		print('UMAP for best reconstruction loss model')
		model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
		val_latent = model.get_latent([adata], batch_size=512, metadata=color_groups)
		figs = tcr.utils.plot_umap_list(val_latent, title=name + '_best_recon', color_groups=color_groups)
		for fig, color_group in zip(figs, color_groups):
			experiment.log_figure(figure_name=name + '_recon_' + color_group, figure=fig, step=model.epoch)

		print('UMAP for best scGen model')
		model.load(os.path.join(save_path, f'{name}_best_model_by_scGen.pt'))
		train_latent = model.get_latent([adata], batch_size=512, metadata=color_groups)
		figs = tcr.utils.plot_umap_list(train_latent, title=name + '_best_scgen', color_groups=color_groups)
		for fig, color_group in zip(figs, color_groups):
			experiment.log_figure(figure_name=name + '_best_scgen' + color_group, figure=fig, step=model.epoch)
		experiment.end()

	return model.best_optimization_metric


utils_train.fix_seeds()
args = utils_train.parse_arguments()


suggest_params = importlib.import_module(f'{args.name}').suggest_params
init_params = importlib.import_module(f'{args.name}').init_params
name = f'{args.name}{args.suffix}'


name += (f'_cond_{args.conditional}' if args.conditional is not None else '')
rna_kld_weight = args.rna_weight


params_fixed = {
	'comet': True,
	'workspace': 'scc-scgen',
	'metadata': ['treatment', 'response', 'patient', 'cluster_tcr', 'cluster', 'clonotype'],
	'validate_every': 1,
	'save_every': 1,
}
params_fixed.update(vars(args))
params_fixed['name'] = name

optimization_params = {
	'column_fold': 'patient',
	'column_perturbation': 'treatment',
	'indicator_perturbation': 'pre',
	'column_cluster': 'cluster'
}


if not os.path.exists(f'../optuna/{name}'):
	os.makedirs(f'../optuna/{name}')

storage_location = f'../optuna/{name}/state.db'
storage_name = f'sqlite:///{storage_location}'
if os.path.exists(storage_location) and not args.resume:  # if not resume
	print('Backup previous experiment database and models')
	os.rename(f'../optuna/{name}', f'../optuna/{name}_{datetime.now().strftime("%Y%m%d-%H.%M")}_backup')
	if not os.path.exists(f'../optuna/{name}'):
		os.makedirs(f'../optuna/{name}')

sampler = optuna.samplers.TPESampler(seed=42)  # Make the sampler behave in a deterministic way.
study = optuna.create_study(study_name=name, sampler=sampler, storage=storage_name, direction='minimize',
							load_if_exists=args.resume)
if not args.resume:
	for param in init_params:
		study.enqueue_trial(param)

study.optimize(objective, n_trials=args.num_samples)
