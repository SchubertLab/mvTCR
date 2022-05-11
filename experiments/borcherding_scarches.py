"""
python -u borcherding_optuna.py --model moe
"""
# comet-ml must be imported before torch and sklearn
import comet_ml

import sys
sys.path.append('..')

import numpy as np
import torch
from tcr_embedding.models.model_selection import run_model_selection
import tcr_embedding.utils_training as utils
from tcr_embedding.utils_preprocessing import group_shuffle_split

import os
import argparse


random_seed = 42
utils.fix_seeds(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--rna_weight', type=int, default=1)
parser.add_argument('--model', type=str, default='moe')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--set', type=int, default=1)
parser.add_argument('--wo_tcr_genes', type=str, default='False')
parser.add_argument('--conditional', type=str, default='Cohort')
parser.add_argument('--embedding', type=bool, default=True)

args = parser.parse_args()

adata = utils.load_data('borcherding')
adata.obs['Tissue+Type'] = [f'{tissue}.{type_}' for tissue, type_ in zip(adata.obs['Tissue'], adata.obs['Type'])]

# Remove some studies from data
if args.set == 1:
    holdout_cohorts = ['GSE154826',  # Lung with normal and tumor cells 18387 cells
                       'GSE121636',  # Renal with 4594 cells
                       'GSE162500',  # Lung with 18850 cells
                       ]
elif args.set == 2:
    holdout_cohorts = ['GSE154826',  # Lung with normal and tumor cells 18387 cells
                       'GSE121636',  # Renal with 4594 cells
                       'GSE139555',  # Mixed with lung, renal, colorectal and encometrical cells
                       ]

adata = adata[~adata.obs[args.conditional].isin(holdout_cohorts)].copy()
adata.obsm[args.conditional] = torch.nn.functional.one_hot(torch.tensor(adata.obs[args.conditional].factorize()[0])).numpy()


train, val = group_shuffle_split(adata, group_col='clonotype', val_split=0.2, random_seed=random_seed)
adata.obs['set'] = 'train'
adata.obs.loc[val.obs.index, 'set'] = 'val'
adata = adata[adata.obs['set'].isin(['train', 'val'])]

if args.wo_tcr_genes == 'True':
    tcr_gene_prefixs = ['TRAV', 'TRAJ', 'TRAC', 'TRB', 'TRDV', 'TRDC', 'TRG']
    non_tcr_genes = adata.var_names
    for prefix in tcr_gene_prefixs:
        non_tcr_genes = [el for el in non_tcr_genes if not el.startswith(prefix)]
    adata = adata[:, non_tcr_genes]

params_experiment = {
    'study_name': f'borcherding_{args.model}_{args.rna_weight}_{args.conditional}_{args.wo_tcr_genes}_emb_{args.embedding}_set_{args.set}',
    'comet_workspace': None,  # 'Covid',
    'model_name': args.model,
    'early_stop': 5,
    'balanced_sampling': 'clonotype',
    'metadata': ['clonotype', 'Sample', 'Type', 'Tissue', 'Tissue+Type', 'functional.cluster'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna', f'borcherding_{args.model}_{args.rna_weight}_{args.conditional}_{args.wo_tcr_genes}_set_{args.set}'),
    'conditional': args.conditional,
    'use_embedding_for_cond': args.embedding  # use one-hot
}
if args.model == 'rna':
    params_experiment['balanced_sampling'] = None

params_optimization = {
    'name': 'pseudo_metric',
    'prediction_labels':
        {'clonotype': 1,
         'Tissue+Type': args.rna_weight}
}

timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)


