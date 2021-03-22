# python -W ignore 10x_ray_tune.py --model bigru --name test
from comet_ml import Experiment
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import scanpy as sc
import os
import pickle
from datetime import datetime
import argparse
from tqdm import tqdm
import time
import importlib
import sys
sys.path.append('../config_tune')

import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch


def trial_dirname_creator(trial):
	return f'{datetime.now().strftime("%Y%m%d_%H-%M-%S")}_{trial.trial_id}'


def objective(params, checkpoint_dir=None, adata=None):
	"""
	Objective function for Ray Tune
	:param params: Ray Tune will use this automatically
	:param checkpoint_dir: Ray Tune will use this automatically
	:param adata: adata containing train and eval
	"""
	import warnings
	warnings.simplefilter(action='ignore', category=FutureWarning)
	import pandas as pd
	pd.options.mode.chained_assignment = None  # default='warn'

	import sys
	sys.path.append('../../../')
	import tcr_embedding as tcr  # tune needs to reload this module
	from tcr_embedding.evaluation.WrapperFunctions import get_model_prediction_function
	from tcr_embedding.evaluation.Imputation import run_imputation_evaluation

	# Change hyperparameters to match our syntax, this includes
	# - Optuna cannot sample within lists, so we have to add those values back into a list
	params['loss_weights'] = [params['loss_weights_scRNA'], params['loss_weights_seq'], params['loss_weights_kl']]
	params['shared_hidden'] = [params['shared_hidden']]

	if 'gene_hidden' in params['scRNA_model_hyperparams']:
		params['scRNA_model_hyperparams']['gene_hidden'] = [params['scRNA_model_hyperparams']['gene_hidden']]

	if params['seq_model_arch'] == 'CNN':
		# Encoder
		params['seq_model_hyperparams']['encoder']['kernel'] = [
			params['seq_model_hyperparams']['encoder']['kernel_1'],
			params['seq_model_hyperparams']['encoder']['kernel_23'],
			params['seq_model_hyperparams']['encoder']['kernel_23']
		]
		params['seq_model_hyperparams']['encoder']['stride'] = [
			params['seq_model_hyperparams']['encoder']['stride_1'],
			params['seq_model_hyperparams']['encoder']['stride_23'],
			params['seq_model_hyperparams']['encoder']['stride_23']
		]
		params['seq_model_hyperparams']['encoder']['num_features'] = [
			params['seq_model_hyperparams']['encoder']['num_features_1'],
			params['seq_model_hyperparams']['encoder']['num_features_2'],
			params['seq_model_hyperparams']['encoder']['num_features_3']
		]
		# Decoder
		params['seq_model_hyperparams']['decoder']['kernel'] = [
			params['seq_model_hyperparams']['decoder']['kernel_1'],
			params['seq_model_hyperparams']['decoder']['kernel_2'],
		]
		params['seq_model_hyperparams']['decoder']['stride'] = [
			params['seq_model_hyperparams']['decoder']['stride_1'],
			params['seq_model_hyperparams']['decoder']['stride_2'],
		]
		params['seq_model_hyperparams']['decoder']['num_features'] = [
			params['seq_model_hyperparams']['decoder']['num_features_1'],
			params['seq_model_hyperparams']['decoder']['num_features_2'],
			params['seq_model_hyperparams']['decoder']['num_features_3']
		]
	# Init Comet-ML
	current_datetime = datetime.now().strftime("%Y%m%d-%H.%M")
	experiment_name = name + '_' + current_datetime
	with open('../../../comet_ml_key/API_key.txt') as f:
		COMET_ML_KEY = f.read()

	experiment = Experiment(api_key=COMET_ML_KEY, workspace='tcr', project_name=name)
	experiment.log_parameters(params)
	experiment.log_parameters(params['scRNA_model_hyperparams'], prefix='scRNA')
	experiment.log_parameters(params['seq_model_hyperparams'], prefix='seq')
	experiment.log_parameter('experiment_name', experiment_name)
	if params['seq_model_arch'] == 'CNN':
		experiment.log_parameters(params['seq_model_hyperparams']['encoder'], prefix='seq_encoder')
		experiment.log_parameters(params['seq_model_hyperparams']['decoder'], prefix='seq_decoder')

	adata = adata[adata.obs['set'] != 'test']  # This needs to be inside the function, ray can't deal with it outside

	if 'single' in args.model:
		init_model = tcr.models.single_model.SingleModel
	else:
		init_model = tcr.models.joint_model.JointModel
	# Init Model
	model = init_model(
		adatas=[adata],  # adatas containing gene expression and TCR-seq
		aa_to_id=adata.uns['aa_to_id'],  # dict {aa_char: id}
		seq_model_arch=params['seq_model_arch'],  # seq model architecture
		seq_model_hyperparams=params['seq_model_hyperparams'],  # dict of seq model hyperparameters
		scRNA_model_arch=params['scRNA_model_arch'],
		scRNA_model_hyperparams=params['scRNA_model_hyperparams'],
		zdim=params['zdim'],  # zdim
		hdim=params['hdim'],  # hidden dimension of scRNA and seq encoders
		activation=params['activation'],  # activation function of autoencoder hidden layers
		dropout=params['dropout'],
		batch_norm=params['batch_norm'],
		shared_hidden=params['shared_hidden'],  # hidden layers of shared encoder / decoder
		names=['10x'],
		gene_layers=[],  # [] or list of str for layer keys of each dataset
		seq_keys=[]  # [] or list of str for seq keys of each dataset
	)

	#     # Code Snippet for later
	#     # Not needed for Optuna, only for Scheduler such as Hyperband, ASHA, PBT
	#     if checkpoint_dir:
	#         model_state, optimizer_state = torch.load(
	#             os.path.join(checkpoint_dir, "checkpoint"))
	#         model.model.load_state_dict(model_state)
	#         optimizer.load_state_dict(optimizer_state)

	#     # Needs to be integrated into the training code, as checkpoint is saved every epoch
	#     with tune.checkpoint_dir(epoch) as checkpoint_dir:
	#         path = os.path.join(checkpoint_dir, "checkpoint")
	#         torch.save((net.state_dict(), optimizer.state_dict()), path)
	with tune.checkpoint_dir(0) as checkpoint_dir:
		save_path = checkpoint_dir

	n_epochs = args.n_epochs * params['batch_size'] // 256  # to have same numbers of iteration
	epoch2step = 256 / params['batch_size']  # normalization factor of epoch -> step, as one epoch with different batch_size results in different numbers of iterations
	epoch2step *= 1000  # to avoid decimal points, as we multiply with a float number
	save_every = n_epochs // args.num_checkpoints
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
		early_stop=args.early_stop,
		validate_every=5,
		save_every=save_every,
		save_path=save_path,
		num_workers=0,
		verbose=0,  # 0: only tdqm progress bar, 1: val loss, 2: train and val loss
		device=None,
		comet=experiment
	)
	# Evaluate each checkpoint model
	best_epoch = -1
	best_metric = -1
	metrics_list = []
	for e in tqdm(range(0, n_epochs+1, save_every), 'kNN for previous checkpoints: '):
		if os.path.exists(os.path.join(save_path, f'{name}_epoch_{str(e).zfill(5)}.pt')):
			model.load(os.path.join(save_path, f'{name}_epoch_{str(e).zfill(5)}.pt'))
			test_embedding_func = get_model_prediction_function(model, batch_size=params['batch_size'])
			try:
				summary = run_imputation_evaluation(adata, test_embedding_func, query_source='val', use_non_binder=True, use_reduced_binders=True)
			except:
				tune.report(weighted_f1=0.0)
				return

			metrics = summary['knn']
			metrics_list.append(metrics['weighted avg']['f1-score'])
			for antigen, metric in metrics.items():
				if antigen != 'accuracy':
					experiment.log_metrics(metric, prefix=antigen, step=int(model.epoch*epoch2step), epoch=model.epoch)
				else:
					experiment.log_metric('accuracy', metric, step=int(model.epoch*epoch2step), epoch=model.epoch)

			if metrics['weighted avg']['f1-score'] > best_metric:
				best_metric = metrics['weighted avg']['f1-score']
				best_epoch = e

	checkpoint_fps = os.listdir(save_path)
	checkpoint_fps = [checkpoint_fp for checkpoint_fp in checkpoint_fps if '_epoch_' in checkpoint_fp]
	checkpoint_fps.remove(f'{name}_epoch_{str(best_epoch).zfill(5)}.pt')
	for checkpoint_fp in checkpoint_fps:
		os.remove(os.path.join(save_path, checkpoint_fp))

	print('kNN for best reconstruction loss model')
	# Evaluate Model (best model based on reconstruction loss)
	model.load(os.path.join(save_path, f'{name}_best_model.pt'))
	test_embedding_func = get_model_prediction_function(model, batch_size=params['batch_size'])
	try:
		summary = run_imputation_evaluation(adata, test_embedding_func, query_source='val', use_non_binder=True, use_reduced_binders=True)
	except:
		tune.report(weighted_f1=0.0)
		return

	metrics = summary['knn']
	metrics_list.append(metrics['weighted avg']['f1-score'])
	for antigen, metric in metrics.items():
		if antigen != 'accuracy':
			experiment.log_metrics(metric, prefix='best_recon_'+antigen, step=int(model.epoch*epoch2step), epoch=model.epoch)
		else:
			experiment.log_metric('best_recon_accuracy', metric, step=int(model.epoch*epoch2step), epoch=model.epoch)

	experiment.end()

	# Report Metric back to Tune
	for metric in metrics_list:
		print(f'Weighted F1: {metric}')
		tune.report(weighted_f1=metric)
		time.sleep(0.5)


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='If flag is set, then resumes previous training')
parser.add_argument('--model', type=str, default='single_scRNA')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--early_stop', type=int, default=100)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--num_checkpoints', type=int, default=20)
parser.add_argument('--local_mode', action='store_true', help='If flag is set, then local mode in ray is activated which enables breakpoints')
parser.add_argument('--num_cpu', type=int, default=4)
parser.add_argument('--num_gpu', type=int, default=1)

args = parser.parse_args()

adata = sc.read_h5ad('../data/10x_CD8TC/v5_train_val_test.h5ad')

params = importlib.import_module(f'{args.model}_tune').params
init_params = importlib.import_module(f'{args.model}_tune').init_params

name = f'10x_tune_{args.model}{args.suffix}'
local_dir = '~/tcr-embedding/ray_results'
ray.init(local_mode=args.local_mode)

algo = OptunaSearch(metric='weighted_f1', mode='max', points_to_evaluate=init_params)
algo = ConcurrencyLimiter(algo, max_concurrent=2)
analysis = tune.run(
	tune.with_parameters(objective, adata=adata),
	name=name,
	metric='weighted_f1',
	mode='max',
	search_alg=algo,
	num_samples=args.num_samples,
	config=params,
	resources_per_trial={'cpu': args.num_cpu, 'gpu': args.num_gpu},
	local_dir=local_dir,
	trial_dirname_creator=trial_dirname_creator,
	verbose=3,
	resume=args.resume
)

with open(os.path.join('../ray_results', name, 'analysis_file.pkl'), mode='wb') as f:
	pickle.dump(analysis, f)

print('Best hyperparameter were: ')
print(analysis.best_config)
