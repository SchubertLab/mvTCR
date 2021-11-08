"""
python -u covid_optuna.py --model poe --split 0
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


utils.fix_seeds(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='poe')
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
args = parser.parse_args()


adata = utils.load_data('covid')

# subsample to get statistics
random_seed = args.split
sub, non_sub = stratified_group_shuffle_split(adata.obs, stratify_col='condition', group_col='clonotype',
                                              val_split=0.2, random_seed=random_seed)
train, val = stratified_group_shuffle_split(sub, stratify_col='condition', group_col='clonotype',
                                            val_split=0.20, random_seed=random_seed)
adata.obs['set'] = 'train'
adata.obs.loc[non_sub.index, 'set'] = '-'
adata.obs.loc[val.index, 'set'] = 'val'
adata = adata[adata.obs['set'].isin(['train', 'val'])]


params_experiment = {
    'study_name': f'Covid_{args.model}_split_{args.split}',
    'comet_workspace': 'Covid',
    'model_name': args.model,
    'balanced_sampling': 'clonotype',
    'metadata': ['identifier', 'cell_type', 'condition', 'responsive', 'reactive_combined'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna',
                              f'Covid_{args.model}_split_{args.split}')
}
if args.model == 'rna':
    params_experiment['balanced_sampling'] = None

params_optimization = {
    'name': 'pseudo_metric',
    'prediction_labels': ['clonotype', 'cell_type'],
}

timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)
