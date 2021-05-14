# python -W ignore 10x_ray_tune.py --model bigru --name test
from comet_ml import Experiment
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import scanpy as sc
import os
import pickle
from datetime import datetime
import argparse
import importlib
import sys
sys.path.append('../config_tune')

import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch

import tcr_embedding.utils_training as helper
import experiments.BCC_scGen_train as sc_gen


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--resume', action='store_true', help='Resumes previous training', default=False)
	parser.add_argument('--model', type=str, default='single_scRNA')
	parser.add_argument('--suffix', type=str, default='')
	parser.add_argument('--n_epochs', type=int, default=5000)
	parser.add_argument('--early_stop', type=int, default=100)
	parser.add_argument('--num_samples', type=int, default=100)
	parser.add_argument('--num_checkpoints', type=int, default=20)
	parser.add_argument('--local_mode', action='store_true', help='Local mode in ray is activated, enables breakpoints')
	parser.add_argument('--num_cpu', type=int, default=4)
	parser.add_argument('--num_gpu', type=int, default=1)
	parser.add_argument('--balanced_sampling', type=str, default=None, help='name of the column used to balance')
	parser.add_argument('--grid_search', action='store_true')
	args = parser.parse_args()
	return args


def trial_dirname_creator(trial):
	"""

	:param trial:
	:return:
	"""
	return f'{datetime.now().strftime("%Y%m%d_%H-%M-%S")}_{trial.trial_id}'


def initialize_libraries(seed=42):
	""" imports and initializes all libraries needed in the objective function
	:param seed: seed number
	"""
	import torch
	import numpy as np
	import random

	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)


def objective(params, params_scgen, params_fixed):
	"""
	Objective function for Ray Tune
	:param params: Ray Tune will use this automatically
	"""
	initialize_libraries(42)

	with tune.checkpoint_dir(0) as checkpoint_dir:
		save_path = checkpoint_dir
		params_fixed['save_path'] = save_path

	comet = helper.initialize_comet(params_hpo, params_fixed)
	comet.log_parameters(params_scgen, prefix='scGen')

	model_type = helper.select_model_by_name(params_fixed.name)

	sc_gen.training_loop(params, params_scgen, params_fixed, model_type, comet)

	comet.end()


fixed_params = parse_arguments()

params_hpo = importlib.import_module(f'{fixed_params.model}_tune').params
init_params = importlib.import_module(f'{fixed_params.model}_tune').init_params

name = f'bcc_tune_{fixed_params.model}{fixed_params.suffix}'

local_dir = os.path.join(os.path.dirname(__file__), '../ray_results')
ray.init(local_mode=fixed_params.local_mode)

if fixed_params['grid_search']:
	algo = None
else:
	algo = OptunaSearch(metric='reconstruction', mode='min', points_to_evaluate=init_params)
	algo = ConcurrencyLimiter(algo, max_concurrent=2)


scgen_params = {
	'pertubation': 'treatment',
	'indicator': 'pre',
	'hold_out': ('patient', 'su009')
}


analysis = tune.run(
	tune.with_parameters(objective, scgen_params=scgen_params, params_fixed=fixed_params),
	name=name,
	metric='scgen',
	mode='max',
	search_alg=algo,
	num_samples=fixed_params.num_samples,
	config=params_hpo,
	resources_per_trial={'cpu': fixed_params.num_cpu, 'gpu': fixed_params.num_gpu},
	local_dir=local_dir,
	trial_dirname_creator=trial_dirname_creator,
	verbose=3,
	resume=fixed_params.resume
)

