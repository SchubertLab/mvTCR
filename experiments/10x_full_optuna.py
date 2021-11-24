"""
python -u 10x_optuna.py --model poe --donor 1 --split 0
"""
# comet-ml must be imported before torch and sklearn
import comet_ml

import sys
sys.path.append('..')

from tcr_embedding.models.model_selection import run_model_selection
import tcr_embedding.utils_training as utils
from tcr_embedding.utils_preprocessing import stratified_group_shuffle_split

import os
import argparse
import config.constants_10x as const


random_seed = 42
utils.fix_seeds(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='poe')
parser.add_argument('--donor', type=str, default=None)
parser.add_argument('--filter_non_binder', type=bool, default=True)
parser.add_argument('--gpus', type=int, default=1)
args = parser.parse_args()


adata = utils.load_data('10x')
if args.donor is not None:
    adata = adata[adata.obs['donor'] == f'donor_{args.donor}']
if args.filter_non_binder:
    adata = adata[adata.obs['binding_name'].isin(const.HIGH_COUNT_ANTIGENS)]


train, val = stratified_group_shuffle_split(adata.obs, stratify_col='binding_name', group_col='clonotype',
                                              val_split=0.2, random_seed=random_seed)
adata.obs['set'] = 'train'
adata.obs.loc[val.index, 'set'] = 'val'


params_experiment = {
    'study_name': f'10x_{args.donor}_{args.model}_filtered_{args.filter_non_binder}_full',
    'comet_workspace': '10x',
    'model_name': args.model,
    'balanced_sampling': 'clonotype',
    'metadata': ['binding_name', 'clonotype', 'donor'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna',
                              f'10x_{args.donor}_{args.model}_full')
}
if args.model == 'rna':
    params_experiment['balanced_sampling'] = None

params_optimization = {
    'name': 'knn_prediction',
    'prediction_column': 'binding_name',
}

timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)
