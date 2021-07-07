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
from tcr_embedding.evaluation.WrapperFunctions import get_model_prediction_function
from tcr_embedding.evaluation.Clustering import run_clustering_evaluation
from tcr_embedding.evaluation.kNN import run_knn_evaluation

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
	if params['seq_model_arch'] == 'CNN':
		experiment.log_parameters(params['seq_model_hyperparams']['encoder'], prefix='seq_encoder')
		experiment.log_parameters(params['seq_model_hyperparams']['decoder'], prefix='seq_decoder')
	experiment.log_parameters(vars(args))

	adata = sc.read_h5ad('../data/Haniffa/v3_conditional.h5ad')
	adata = adata[adata.obs['set'] != 'test']  # This needs to be inside the function, ray can't deal with it outside

	model = init_model(params, model_type=args.model, adata=adata, dataset_name='haniffa', use_cov=args.use_cov, conditional=args.conditional)
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
		metadata=None,
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

	color_groups = ['Site', 'patient_id', 'Sex', 'full_clustering', 'clonotype', 'initial_clustering', 'Swab_result',
					'Status',  'Status_on_day_collection_summary', 'Worst_Clinical_Status', 'Outcome']
	if os.path.exists(os.path.join(save_path, f'{name}_best_rec_model.pt')):
		# UMAP
		print('UMAP for best reconstruction loss model on val')
		model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
		val_latent = model.get_latent([adata[adata.obs['set'] == 'val']], batch_size=512, metadata=color_groups)
		figs = tcr.utils.plot_umap_list(val_latent, title=name + '_val_best_recon', color_groups=color_groups)
		for fig, color_group in zip(figs, color_groups):
			experiment.log_figure(figure_name=name + '_val_recon_' + color_group, figure=fig, step=model.epoch)

		print('UMAP for best reconstruction loss model on train')
		model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
		train_latent = model.get_latent([adata[adata.obs['set'] == 'train']], batch_size=512, metadata=color_groups)
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

		print('kNN evaluation')
		summary = run_knn_evaluation(adata, test_embedding_func, name_label='full_clustering', query_source='val', num_neighbors=5)
		metrics = summary['knn']
		for key, metric in metrics.items():
			if key != 'accuracy':
				experiment.log_metrics(metric, prefix=f'Val {key}', epoch=model.epoch)
			else:
				experiment.log_metric('Val accuracy', metric, epoch=model.epoch)

		experiment.end()

	return model.best_loss


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='If flag is set, then resumes previous training')
parser.add_argument('--model', type=str, default='RNA')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--early_stop', type=int, default=100)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--balanced_sampling', type=str, default=None)
parser.add_argument('--conditional', type=str, default=None)
parser.add_argument('--use_cov', action='store_true', help='If flag is set, CoV-weighting is used')
args = parser.parse_args()

if args.name is not None:
	suggest_params = importlib.import_module(f'{args.name}').suggest_params
	init_params = importlib.import_module(f'{args.name}').init_params
	name = f'{args.name}'
else:
	suggest_params = importlib.import_module(f'haniffa_{args.model.lower()}').suggest_params
	init_params = importlib.import_module(f'haniffa_{args.model.lower()}').init_params
	name = f'haniffa_{args.model}{args.suffix}'

name = name + ('_CoV' if args.use_cov else '') + (f'_cond_{args.conditional}' if args.conditional is not None else '')
if not os.path.exists(f'../optuna/{name}'):
	os.makedirs(f'../optuna/{name}')

storage_location = f'../optuna/{name}/state.db'
storage_name = f'sqlite:///{storage_location}'
if os.path.exists(storage_location) and not args.resume:  # if not resume
	print('Backup previous experiment database')
	os.rename(storage_location, f'../optuna/{name}/state_{datetime.now().strftime("%Y%m%d-%H.%M")}_backup.db')

sampler = optuna.samplers.TPESampler(seed=random_seed)  # Make the sampler behave in a deterministic way.
study = optuna.create_study(study_name=name, sampler=sampler, storage=storage_name, direction='minimize', load_if_exists=args.resume)
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

