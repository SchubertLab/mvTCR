"""
python -u yost_optuna_cell_type.py --model moe --cell_type CD4_T_cells --data bcc
bcc ['CD4_T_cells', 'CD8_mem_T_cells', 'Tregs', 'CD8_act_T_cells', 'CD8_ex_T_cells', 'Tcell_prolif']
scc ['Th17', 'CD8_naive', 'CD8_ex', 'Naive', 'CD8_mem', 'Treg', 'Tfh', 'CD8_act', 'CD8_eff', 'CD8_ex_act']
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
parser.add_argument('--cell_type', type=str, default=None)
parser.add_argument('--data', type=str, default='bcc')
parser.add_argument('--gpus', type=int, default=1)
args = parser.parse_args()


adata = utils.load_data(args.data)

if args.cell_type is not None:
    adata = adata[adata.obs['cluster'] != args.cell_type]


random_seed = 42
train, val = group_shuffle_split(adata, group_col='clonotype', val_split=0.25, random_seed=random_seed)
adata.obs['set'] = 'train'
adata.obs.loc[val.obs.index, 'set'] = 'val'
adata.obs['set'] = adata.obs['set'].astype('category')


params_experiment = {
    'study_name': f'scGen_ct_{args.data}_{args.cell_type}_{args.model}',
    'comet_workspace': None,
    'model_name': args.model,
    'balanced_sampling': 'clonotype',
    'metadata': [],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna',
                              f'scGen_ct_{args.data}_{args.cell_type}_{args.model}')
}

if args.model == 'rna':
    params_experiment['balanced_sampling'] = None

sc.tl.rank_genes_groups(adata, 'cluster', n_genes=100, method='wilcoxon')
degs = adata.uns['rank_genes_groups']['names']
degs = [j for i in degs for j in i]

params_optimization = {
    'name': 'modulation_prediction',
    'column_fold': 'cluster',
    'column_perturbation': 'treatment',
    'indicator_perturbation': 'pre',
    'gene_set': degs,
}

timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)
