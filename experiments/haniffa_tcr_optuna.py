"""
python -u haniffa_tcr_optuna.py --model moe
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


utils.fix_seeds(42)

parser = argparse.ArgumentParser()
parser.add_argument('--rna_weight', type=int, default=1)
parser.add_argument('--model', type=str, default='moe')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--wo_tcr_genes', type=str, default="False")
parser.add_argument('--conditional', type=str, default='patient_id')
parser.add_argument('--site', type=str, default=None)

args = parser.parse_args()


adata = utils.load_data('haniffa')

if args.wo_tcr_genes == 'True':
    tcr_gene_prefixs = ['TRAV', 'TRAJ', 'TRAC', 'TRB', 'TRDV', 'TRDC', 'TRG']
    non_tcr_genes = adata.var_names
    for prefix in tcr_gene_prefixs:
        non_tcr_genes = [el for el in non_tcr_genes if not el.startswith(prefix)]
    adata = adata[:, non_tcr_genes]

if args.site is not None:
    adata = adata[adata.obs['Site'] == args.site]

# subsample to get statistics
random_seed = 42
train, val = group_shuffle_split(adata, group_col='clonotype', val_split=0.20, random_seed=random_seed)
adata.obs['set'] = 'train'
adata.obs.loc[val.obs.index, 'set'] = 'val'
adata = adata[adata.obs['set'].isin(['train', 'val'])]


params_experiment = {
    'study_name': f'haniffa_tcr_{args.model}_{args.rna_weight}_{args.conditional}_{args.wo_tcr_genes}_{args.site}',
    'comet_workspace': None,  # 'Covid',
    'model_name': args.model,
    'balanced_sampling': 'clonotype',
    'metadata': ['clonotype', 'full_clustering'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna',
                              f'haniffa_tcr_{args.model}_{args.rna_weight}_{args.conditional}_{args.wo_tcr_genes}'
                              + f'{args.site}'),
    'conditional': args.conditional
}
if args.model == 'rna':
    params_experiment['balanced_sampling'] = None

params_optimization = {
    'name': 'pseudo_metric',
    'prediction_labels':
        {'clonotype': 1,
         'full_clustering': args.rna_weight}
}

timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)
