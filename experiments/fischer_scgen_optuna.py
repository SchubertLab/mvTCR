"""
python -u fischer_scgen_optuna.py --model moe
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
parser.add_argument('--model', type=str, default='moe')
parser.add_argument('--gpus', type=int, default=1)
args = parser.parse_args()


adata = utils.load_data('covid')

random_seed = 42
train, val = group_shuffle_split(adata, group_col='clonotype', val_split=0.25, random_seed=random_seed)
adata.obs['set'] = 'train'
adata.obs.loc[val.obs.index, 'set'] = 'val'
adata.obs['set'] = adata.obs['set'].astype('category')


params_experiment = {
    'study_name': f'scGenFischer_{args.model}',
    'comet_workspace': None,
    'model_name': args.model,
    'balanced_sampling': 'clonotype',
    'metadata': [],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna',
                              f'scGenFischer_{args.model}')
}

if args.model == 'rna':
    params_experiment['balanced_sampling'] = None


sc.tl.rank_genes_groups(adata, 'condition', n_genes=50, method='wilcoxon')
degs = adata.uns['rank_genes_groups']['names']
degs = [j for i in degs for j in i]

params_optimization = {
    'name': 'modulation_prediction',
    'column_fold': 'clonotype',
    'column_perturbation': 'condition',
    'indicator_perturbation': 'unstimulated',
    'gene_set': degs,
}


timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)
