"""
python -u 10x_optuna.py --model poe --donor 1 --split 0
"""
# comet-ml must be imported before torch and sklearn
import comet_ml

import sys
sys.path.append('..')

from tcr_embedding.models.model_selection import run_model_selection
import tcr_embedding.utils_training as utils
from tcr_embedding.utils_preprocessing import group_shuffle_split

import os
import argparse

import scanpy as sc


utils.fix_seeds(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='moe')
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=None)
args = parser.parse_args()


adata = utils.load_data('minervina/01_annotated_data.h5ad')
adata.obs['epitope_label'] = adata.obs['epitope'].factorize()[0]

# subsample to get statistics
random_seed = args.split
train_val, test = group_shuffle_split(adata, group_col='clonotype', val_split=0.20, random_seed=random_seed)
train, val = group_shuffle_split(train_val, group_col='clonotype', val_split=0.25, random_seed=random_seed)

if args.n_samples is not None:
    sc.pp.subsample(train, n_obs=args.n_samples)

adata.obs['set'] = None
adata.obs.loc[train.obs.index, 'set'] = 'train'
adata.obs.loc[val.obs.index, 'set'] = 'val'
adata.obs.loc[test.obs.index, 'set'] = 'test'
adata = adata[adata.obs['set'].isin(['train', 'val'])]

n_samples_prefix = '' if args.n_samples is None else f'_{args.n_samples}'

params_experiment = {
    'study_name': f'minervina_{args.model}_split_{args.split}{n_samples_prefix}_supervised',
    'comet_workspace': None,
    'model_name': args.model + '_supervised',
    'label_key': 'epitope_label',
    'balanced_sampling': 'clonotype',
    'metadata': [],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna',
                              f'minervina_{args.model}_split_{args.split}{n_samples_prefix}_supervised'),
    'conditional': 'donor'
}

if args.model == 'rna':
    params_experiment['balanced_sampling'] = None

params_optimization = {
    'name': 'supervised',
    'prediction_column': 'epitope_label',
}

timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)
