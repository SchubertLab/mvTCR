"""
python -u prediction_p1d.py --model poe --clonotype 5916 --data bcc

top_10_clonotypes = [5916, 5933, 7675, 5910, 2117, 5912, 5977, 4746, 5909, 7660]
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

sc.settings.verbosity = 0

utils.fix_seeds(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='poe')
parser.add_argument('--clonotype', type=int, default=None)
parser.add_argument('--data', type=str, default='bcc')
parser.add_argument('--gpus', type=int, default=1)
args = parser.parse_args()


adata = utils.load_data(args.data)
if args.clonotype is not None:
    adata = adata[adata.obs['clonotype'] != args.clonotype]

random_seed = 42
train, val = group_shuffle_split(adata, group_col='clonotype', val_split=0.25, random_seed=random_seed)
adata.obs['set'] = 'train'
adata.obs.loc[val.obs.index, 'set'] = 'val'
adata.obs['set'] = adata.obs['set'].astype('category')


params_experiment = {
    'study_name': f'scGen_rev_{args.data}_{args.clonotype}_{args.model}',
    'comet_workspace': None,
    'model_name': args.model,
    'balanced_sampling': 'clonotype',
    'metadata': ['patient', 'treatment', 'cluster', 'clonotype', 'response'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna',
                              f'scGen_rev_{args.data}_{args.clonotype}_{args.model}')
}

if args.model == 'rna':
    params_experiment['balanced_sampling'] = None

params_optimization = {
    'name': 'modulation_prediction',
    'column_fold': 'clonotype',
    'column_perturbation': 'treatment',
    'indicator_perturbation': 'pre',
}

timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)
