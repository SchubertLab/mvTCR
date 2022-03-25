"""
python -u 10x_avidity.py --model moe --donor 1 --split 0
"""
# comet-ml must be imported before torch and sklearn
import comet_ml

import sys
sys.path.append('..')

from tcr_embedding.models.model_selection_count_prediction import run_model_selection
import tcr_embedding.utils_training as utils
from tcr_embedding.utils_preprocessing import group_shuffle_split

import os
import argparse
import config.constants_10x as const


utils.fix_seeds(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='poe')
parser.add_argument('--donor', type=str, default=None)
parser.add_argument('--filter_non_binder', type=bool, default=True)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
args = parser.parse_args()


adata = utils.load_data('10x')
if args.donor is not None:
    adata = adata[adata.obs['donor'] == f'donor_{args.donor}']
if args.filter_non_binder:
    adata = adata[adata.obs['binding_name'].isin(const.HIGH_COUNT_ANTIGENS)]

# Add Avidity information



# subsample to get statistics
random_seed = args.split
train_val, test = group_shuffle_split(adata, group_col='clonotype', val_split=0.20, random_seed=random_seed)
train, val = group_shuffle_split(train_val, group_col='clonotype', val_split=0.25, random_seed=random_seed)

adata.obs['set'] = 'train'
adata.obs.loc[val.obs.index, 'set'] = 'val'
adata.obs.loc[test.obs.index, 'set'] = 'test'
adata = adata[adata.obs['set'].isin(['train', 'val'])]


path_model = f'saved_models/journal/10x/splits/donor_{args.donor}/{args.model}/'
path_model += f'{args.model}_donor_{args.donor}_split_{args.split}.pt'

params_experiment = {
    'study_name': f'10x_avidity_{args.donor}_{args.model}_split_{args.split}',
    'model_path': path_model,
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna',
                              f'10x_avidity_{args.donor}_{args.model}_split_{args.split}'),
    'key_prediction': 'binding_counts'
}


timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, 100, timeout, args.gpus)
