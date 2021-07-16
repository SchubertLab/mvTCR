# python 10x_optuna.py --model poe_transformer_d1_without --n_epochs 100 --donor 1 --balanced_sampling clonotype --without_non_binder
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
from tcr_embedding.utils_training import init_model
from tcr_embedding.evaluation.Imputation import run_imputation_evaluation
from tcr_embedding.evaluation.WrapperFunctions import get_model_prediction_function

random_seed = 42
import torch
import numpy as np
import random

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

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

	# Init Comet-ML
	with open('../comet_ml_key/API_key.txt') as f:
		COMET_ML_KEY = f.read()
	experiment = Experiment(api_key=COMET_ML_KEY, workspace='tcr', project_name=name)
	experiment.log_parameters(params)
	experiment.log_parameters(params['scRNA_model_hyperparams'], prefix='scRNA')
	experiment.log_parameters(params['seq_model_hyperparams'], prefix='seq')
	experiment.log_parameter('experiment_name', name + f'_trial_{trial.number}')
	experiment.log_parameter('save_path', save_path)
	experiment.log_parameter('balanced_sampling', args.balanced_sampling)
	experiment.log_parameter('without_non_binder', args.without_non_binder)
	if params['seq_model_arch'] == 'CNN':
		experiment.log_parameters(params['seq_model_hyperparams']['encoder'], prefix='seq_encoder')
		experiment.log_parameters(params['seq_model_hyperparams']['decoder'], prefix='seq_decoder')
	experiment.log_parameters(vars(args))

	adata = sc.read_h5ad('../data/10x_CD8TC/v6_supervised.h5ad')
	# adata = adata[adata.obs['set'] != 'test']  # This needs to be inside the function, ray can't deal with it outside
	adata.obs['binding_name'] = adata.obs['binding_name'].astype(str)

	if args.donor != 'all':
		adata = adata[adata.obs['donor'] == 'donor_'+args.donor]
	if args.without_non_binder:
		# Filter out no_data and rare classes, but only from training set
		adata = adata[(adata.obs['binding_name'].isin(tcr.constants.DONOR_SPECIFIC_ANTIGENS[args.donor]))]
	experiment.log_parameter('donors', adata.obs['donor'].unique().astype(str))

	model = init_model(params, model_type=args.model, adata=adata, dataset_name='10x', use_cov=args.use_cov,
					   conditional=args.conditional, rna_priority=args.rna_priority)
	n_epochs = args.n_epochs * params['batch_size'] // 256  # adjust that different batch_size still have same number of epochs
	early_stop = args.early_stop * params['batch_size'] // 256
	epoch2step = 256 / params['batch_size']  # normalization factor of epoch -> step, as one epoch with different batch_size results in different numbers of iterations
	epoch2step *= 1000  # to avoid decimal points, as we multiply with a float number

	# Train Model
	model.train(
		experiment_name=name,
		n_iters=None,
		n_epochs=n_epochs,
		batch_size=params['batch_size'],
		lr=params['lr'],
		losses=params['losses'],  # list of losses for each modality: losses[0] := scRNA, losses[1] := TCR
		loss_weights=params['loss_weights'], # [] or list of floats storing weighting of loss in order [scRNA, TCR, KLD]
		kl_annealing_epochs=None,
		val_split='set',  # float or str, if float: split is determined automatically, if str: used as key for train-val column
		metadata=['binding_name', 'binding_label'],
		early_stop=early_stop,
		balanced_sampling=args.balanced_sampling,
		validate_every=1,
		save_every=1,
		save_path=save_path,
		save_last_model=False,
		num_workers=0,
		device=None,
		comet=experiment
	)

	# For visualization purpose, we set all rare specificities to no_data
	adata.obs['binding_label'][~adata.obs['binding_name'].isin(tcr.constants.HIGH_COUNT_ANTIGENS)] = -1
	adata.obs['binding_name'][~adata.obs['binding_name'].isin(tcr.constants.HIGH_COUNT_ANTIGENS)] = 'no_data'
	# For visualization purpose, else the scanpy plot script thinks the rare specificities are still there and the colors get skewed
	adata.obs['binding_name'] = adata.obs['binding_name'].astype(str)

	if os.path.exists(os.path.join(save_path, f'{name}_best_knn_model.pt')) and os.path.exists(os.path.join(save_path, f'{name}_best_rec_model.pt')):
		# Val UMAP
		print('UMAP for best f1 score model on val')
		model.load(os.path.join(save_path, f'{name}_best_knn_model.pt'))
		val_latent = model.get_latent([adata[adata.obs['set'] == 'val']], batch_size=512, metadata=['binding_name', 'clonotype', 'donor'])
		fig_donor, fig_clonotype, fig_antigen = tcr.utils.plot_umap(val_latent, title=name+'_val_best_f1')
		experiment.log_figure(figure_name=name+'_val_best_f1_donor', figure=fig_donor, step=model.epoch)
		experiment.log_figure(figure_name=name+'_val_best_f1_clonotype', figure=fig_clonotype, step=model.epoch)
		experiment.log_figure(figure_name=name+'_val_best_f1_antigen', figure=fig_antigen, step=model.epoch)

		print('UMAP for best reconstruction loss model on val')
		model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
		val_latent = model.get_latent([adata[adata.obs['set'] == 'val']], batch_size=512, metadata=['binding_name', 'clonotype', 'donor'])
		fig_donor, fig_clonotype, fig_antigen = tcr.utils.plot_umap(val_latent, title=name + '_val_best_recon')
		experiment.log_figure(figure_name=name + '_val_best_recon_donor', figure=fig_donor, step=model.epoch)
		experiment.log_figure(figure_name=name + '_val_best_recon_clonotype', figure=fig_clonotype, step=model.epoch)
		experiment.log_figure(figure_name=name + '_val_best_recon_antigen', figure=fig_antigen, step=model.epoch)

		# Train UMAP
		print('UMAP for best f1 score model on train')
		model.load(os.path.join(save_path, f'{name}_best_knn_model.pt'))
		val_latent = model.get_latent([adata[adata.obs['set'] == 'train']], batch_size=512, metadata=['binding_name', 'clonotype', 'donor'])
		fig_donor, fig_clonotype, fig_antigen = tcr.utils.plot_umap(val_latent, title=name+'_train_best_f1')
		experiment.log_figure(figure_name=name+'_train_best_f1_donor', figure=fig_donor, step=model.epoch)
		experiment.log_figure(figure_name=name+'_train_best_f1_clonotype', figure=fig_clonotype, step=model.epoch)
		experiment.log_figure(figure_name=name+'_train_best_f1_antigen', figure=fig_antigen, step=model.epoch)

		print('UMAP for best reconstruction loss model on train')
		model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
		val_latent = model.get_latent([adata[adata.obs['set'] == 'train']], batch_size=512, metadata=['binding_name', 'clonotype', 'donor'])
		fig_donor, fig_clonotype, fig_antigen = tcr.utils.plot_umap(val_latent, title=name + '_train_best_recon')
		experiment.log_figure(figure_name=name + '_train_best_recon_donor', figure=fig_donor, step=model.epoch)
		experiment.log_figure(figure_name=name + '_train_best_recon_clonotype', figure=fig_clonotype, step=model.epoch)
		experiment.log_figure(figure_name=name + '_train_best_recon_antigen', figure=fig_antigen, step=model.epoch)

		##################### EVALUATE KNN ON TEST #######################
		model.load(os.path.join(save_path, f'{name}_best_knn_model.pt'))
		test_embedding_func = get_model_prediction_function(model, batch_size=1024)  # helper function for evaluation functions

		# kNN antigen specificity imputation
		summary = run_imputation_evaluation(adata, test_embedding_func, query_source='test', use_non_binder=False,
											use_reduced_binders=True, num_neighbors=5)
		metrics = summary['knn']
		for antigen, metric in metrics.items():
			if antigen != 'accuracy':
				experiment.log_metrics(metric, prefix='test '+antigen, step=-1, epoch=-1)
			else:
				experiment.log_metric('test accuracy', metric, step=-1, epoch=-1)

	else:
		print('Models not saved')

	experiment.log_metric('best weighted avg f1-score', model.best_knn_metric)
	experiment.end()

	# in case kNN failed from the beginning
	if model.best_knn_metric == -1:
		return None

	return model.best_knn_metric


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='If flag is set, then resumes previous training')
parser.add_argument('--model', type=str, default='RNA')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--early_stop', type=int, default=100)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--balanced_sampling', type=str, default=None)
parser.add_argument('--donor', type=str, default='all', choices=['all', '1', '2', '3', '4'])
parser.add_argument('--without_non_binder', action='store_true')
parser.add_argument('--conditional', type=str, default=None)
parser.add_argument('--use_cov', action='store_true', help='If flag is set, CoV-weighting is used')
parser.add_argument('--rna_priority', type=int, default=None, help='If is not none, TCR is only trained every n iter')
args = parser.parse_args()

if args.name is not None:
	suggest_params = importlib.import_module(f'{args.name}').suggest_params
	init_params = importlib.import_module(f'{args.name}').init_params
	name = f'{args.name}{args.suffix}'
else:
	suggest_params = importlib.import_module(f'10x_{args.model.lower()}').suggest_params
	init_params = importlib.import_module(f'10x_{args.model.lower()}').init_params
	name = f'10x_{args.model}{args.suffix}'

name = name + ('_CoV' if args.use_cov else '') + (f'_cond_{args.conditional}' if args.conditional is not None else '')
name += (f'_RnaPriority_{args.rna_priority}' if args.rna_priority is not None else '')

if not os.path.exists(f'../optuna/{name}'):
	os.makedirs(f'../optuna/{name}')

storage_location = f'../optuna/{name}/state.db'
storage_name = f'sqlite:///{storage_location}'
if os.path.exists(storage_location) and not args.resume:  # if not resume
	print('Backup previous experiment database and models')
	os.rename(f'../optuna/{name}', f'../optuna/{name}_{datetime.now().strftime("%Y%m%d-%H.%M")}_backup')
	if not os.path.exists(f'../optuna/{name}'):
		os.makedirs(f'../optuna/{name}')

sampler = optuna.samplers.TPESampler(seed=random_seed)  # Make the sampler behave in a deterministic way.
study = optuna.create_study(study_name=name, sampler=sampler, storage=storage_name, direction='maximize', load_if_exists=args.resume)
if not args.resume:
	for param in init_params:
		study.enqueue_trial(param)

study.optimize(objective, n_trials=args.num_samples)

pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
	print("    {}: {}".format(key, value))

