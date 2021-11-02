from tcr_embedding.models.model_selection import run_model_selection
import tcr_embedding.utils_training as utils

import os
import argparse

import sys

sys.path.append('..')
import config.constants_10x as const

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='poe')
parser.add_argument('--donor', type=str, default=None)
parser.add_argument('--filter_non_binder', type=bool, default=True)
parser.add_argument('--split', type=int, default=0)
args = parser.parse_args()

# todo splitting
adata = utils.load_data('10x')
adata = adata[adata.obs['set'] != 'test']
if args.donor is not None:
    adata = adata[adata.obs['donor'] == args.donor]
if args.filter_non_binder:
    adata = adata[adata.obs['binding_name'].isin(const.HIGH_COUNT_ANTIGENS)]

params_experiment = {
    'model_name': args.model_name,
    'balanced_sampling': 'clonotype',
    'metadata': ['binding_name', 'clonotype', 'donor'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna',
                              f'10x_{args.donor}_split_{args.split}')
}
if args.model_name == 'rna':
    params_experiment['balanced_sampling'] = None

params_optimization = {
    'name': 'knn_prediction',
    'prediction_column': 'binding_name',
}

study_name = f'10x_{args.donor}_filtered_{args.filter_non_binder}_split_{args.split}'
timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, study_name, 100, timeout)

# adata.obs['binding_name'] = adata.obs['binding_name'].astype(str) potentially needed?
