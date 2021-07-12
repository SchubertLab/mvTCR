# python 10x_optuna.py --model poe_transformer_d1_without --n_epochs 100 --donor 1 --balanced_sampling clonotype --without_non_binder
from comet_ml import Experiment
import scanpy as sc
import os
import argparse
import importlib
import sys
import warnings
from datetime import datetime
import optuna
import logging

sys.path.append('../')
sys.path.append('../config_optuna')
import tcr_embedding as tcr  # tune needs to reload this module
import tcr_embedding.utils_training as utils_train

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
	params_hpo = suggest_params(trial)

	save_path = f'../optuna/{name}/trial_{trial.number}'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	torch.save(params_hpo, os.path.join(save_path, 'params.pkl'))

	# Init Comet-ML
	experiment = utils_train.initialize_comet(params_fixed, None)

	adata = utils_train.load_data('covid')
	adata = adata[adata.obs['set'] != 'test']  # This needs to be inside the function, ray can't deal with it outside

	init_model = utils_train.select_model_by_name(params_fixed['model'])
	model = utils_train.init_model(params_hpo, init_model, adata, 'covid')
	utils_train.train_call(model, params_hpo, params_fixed, experiment)

	if os.path.exists(os.path.join(save_path, f'{name}_best_rec_model.pt')):
		modes = []

		print('UMAP for best reconstruction loss model on val')
		model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
		val_latent = model.get_latent([adata[adata.obs['set'] == 'val']], batch_size=512, metadata=modes)
		figs = tcr.utils.plot_umap_list(val_latent, title=name + '_val_best_recon', color_groups=modes)
		for mode, fig in zip(modes, figs):
			experiment.log_figure(figure_name=name + '_val_best_recon_'+mode, figure=fig, step=model.epoch)

		print('UMAP for best reconstruction loss model on train')
		model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
		val_latent = model.get_latent([adata[adata.obs['set'] == 'train']], batch_size=512, metadata=['binding_name', 'clonotype', 'donor'])
		figs = tcr.utils.plot_umap_list(val_latent, title=name + '_val_best_recon', color_groups=modes)
		for mode, fig in zip(modes, figs):
			experiment.log_figure(figure_name=name + '_val_best_recon_' + mode, figure=fig, step=model.epoch)

	experiment.end()
	return model.best_loss


params_fixed = utils_train.parse_arguments()

suggest_params = importlib.import_module(f'{params_fixed.model}_optuna').suggest_params
init_params = importlib.import_module(f'{params_fixed.model}_optuna').init_params

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
name = f'Covid_optuna_{params_fixed.model}{params_fixed.suffix}'

if not os.path.exists('../optuna/'):
	os.makedirs('../optuna/')

if os.path.exists('../optuna/' + name + '.db') and not params_fixed.resume:  # if not resume
	print('Backup previous experiment database')
	os.rename('../optuna/'+name+'.db', '../optuna/'+name+f'_{datetime.now().strftime("%Y%m%d-%H.%M")}_backup.db')

storage_name = "sqlite:///../optuna/{}.db".format(name)
sampler = optuna.samplers.TPESampler(seed=random_seed)  # Make the sampler behave in a deterministic way.
study = optuna.create_study(study_name=name, sampler=sampler, storage=storage_name,
							direction='minimize', load_if_exists=params_fixed.resume)
if not params_fixed.resume:
	for param in init_params:
		study.enqueue_trial(param)

study.optimize(objective, n_trials=params_fixed.num_samples)

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

