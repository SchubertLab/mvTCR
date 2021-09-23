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
from tcr_embedding.evaluation.kNN import run_knn_within_set_evaluation

import tcr_embedding.utils_training as utils_train

import torch
import numpy as np
import random


warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def objective(trial):
	"""
	Objective function for Optuna
	:param trial: Optuna Trial Object
	"""
	params = suggest_params(trial)

	if rna_kld_weight is not None:
		params['loss_weights'].append(rna_kld_weight)

	save_path = f'../optuna/{name}/trial_{trial.number}'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	torch.save(params, os.path.join(save_path, 'params.pkl'))
	params_fixed['save_path'] = save_path

	experiment = utils_train.initialize_comet(params, params_fixed)

	adata = utils_train.load_data('haniffa')
	adata = adata[adata.obs['set'] != 'test']  # This needs to be inside the function, ray can't deal with it outside

	model = utils_train.init_model(params, model_type=params_fixed['model'], adata=adata,
								   dataset_name='haniffa', conditional=params_fixed['conditional'],
								   optimization_mode='PseudoMetric', optimization_mode_params=optimization_params)

	utils_train.train_call(model, params, params_fixed, experiment)

	color_groups = ['Site', 'patient_id', 'Sex', 'full_clustering', 'clonotype', 'initial_clustering', 'Swab_result',
					'Status',  'Status_on_day_collection_summary', 'Worst_Clinical_Status', 'Outcome']
	if os.path.exists(os.path.join(save_path, f'{name}_best_rec_model.pt')):
		# UMAP
		print('UMAP for best reconstruction loss model on val')
		model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
		val_latent = model.get_latent([adata[adata.obs['set'] == 'val']], batch_size=1024, metadata=color_groups)
		figs = tcr.utils.plot_umap_list(val_latent, title=name + '_val_best_recon', color_groups=color_groups)
		for fig, color_group in zip(figs, color_groups):
			experiment.log_figure(figure_name=name + '_val_recon_' + color_group, figure=fig, step=model.epoch)

		print('UMAP for best reconstruction loss model on train')
		model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
		train_latent = model.get_latent([adata[adata.obs['set'] == 'train']], batch_size=1024, metadata=color_groups)
		figs = tcr.utils.plot_umap_list(train_latent, title=name + '_train_best_recon', color_groups=color_groups)
		for fig, color_group in zip(figs, color_groups):
			experiment.log_figure(figure_name=name + '_train_recon_' + color_group, figure=fig, step=model.epoch)

		print('Cluster score evaluation')
		test_embedding_func = get_model_prediction_function(model, batch_size=1024)  # helper function for evaluation function
		cluster_results = []
		for resolution in [0.01, 0.1, 1.0]:
			cluster_result = run_clustering_evaluation(adata, test_embedding_func, 'train', name_label='full_clustering',
													   cluster_params={'resolution': resolution, 'num_neighbors': 15}, visualize=False)
			cluster_results.append(cluster_result)

		cluster_results = pd.DataFrame(cluster_results)
		# weighted sum, since ASW goes from -1 to 1, while NMI goes from 0 to 1
		cluster_results['weighted_sum'] = 0.5 * cluster_results['ASW'] + cluster_results['NMI']
		idxmax = cluster_results['weighted_sum'].idxmax()
		best_cluster_score = cluster_results.loc[idxmax]
		experiment.log_metrics({'ASW': best_cluster_score['ASW'],
								'NMI': best_cluster_score['NMI']},
							   epoch=model.epoch)
		experiment.end()

	return model.best_optimization_metric


utils_train.fix_seeds()
args = utils_train.parse_arguments()

optimization_params = {
	'prediction_labels': ['full_clustering', 'clonotype']
}


suggest_params = importlib.import_module(f'{args.name}').suggest_params
init_params = importlib.import_module(f'{args.name}').init_params
name = f'{args.name}{args.suffix}'


name += (f'_cond_{args.conditional}' if args.conditional is not None else '')
rna_kld_weight = args.rna_weight


params_fixed = {
	'comet': True,
	'workspace': 'haniffa2',
	'metadata': ['Site', 'patient_id', 'Sex', 'full_clustering', 'clonotype', 'initial_clustering', 'Swab_result',
					'Status',  'Status_on_day_collection_summary', 'Worst_Clinical_Status', 'Outcome'],
	'validate_every': 1,
	'save_every': 1,
}
params_fixed.update(vars(args))
params_fixed['name'] = name


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
study = optuna.create_study(study_name=name, sampler=sampler, storage=storage_name, direction='maximize',
							load_if_exists=args.resume)
if not args.resume:
	for param in init_params:
		study.enqueue_trial(param)

study.optimize(objective, n_trials=args.num_samples)
