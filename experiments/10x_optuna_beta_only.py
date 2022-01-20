"""
python -u 10x_optuna_beta_only.py --donor 1 --split 0
to compare our model to tessa, we will define a clonotype on base of its cdr3beta alone and train solely on CDR3b
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
import config.constants_10x as const


utils.fix_seeds(42)

parser = argparse.ArgumentParser()
parser.add_argument('--donor', type=str, default=None)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
args = parser.parse_args()


adata = utils.load_data('10x')
if args.donor is not None:
    adata = adata[adata.obs['donor'] == f'donor_{args.donor}']

adata = adata[adata.obs['binding_name'].isin(const.HIGH_COUNT_ANTIGENS)]


# subsample to get statistics
random_seed = args.split

adata.obs['group_col'] = [seq[1:-1] for seq in adata.obs['IR_VDJ_1_cdr3']]
train_val, test = group_shuffle_split(adata, group_col='group_col', val_split=0.20, random_seed=random_seed)
train, val = group_shuffle_split(train_val, group_col='group_col', val_split=0.25, random_seed=random_seed)


adata.obs['set'] = 'train'
adata.obs.loc[val.obs.index, 'set'] = 'val'
adata.obs.loc[test.obs.index, 'set'] = 'test'
adata = adata[adata.obs['set'].isin(['train', 'val'])]


params_experiment = {
    'study_name': f'10x_{args.donor}_moe_beta_split_{args.split}',
    'comet_workspace': None,
    'model_name': 'moe',
    'balanced_sampling': 'group_col',
    'metadata': ['binding_name', 'group_col'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna',
                              f'10x_{args.donor}_moe_beta_split_{args.split}'),
    'beta_only': True,
}

params_optimization = {
    'name': 'knn_prediction',
    'prediction_column': 'binding_name',
}

timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)
