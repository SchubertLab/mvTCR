"""
python -u borcherding.py --gpus 1
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


random_seed = 42
utils.fix_seeds(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=int, default=1)
args = parser.parse_args()


adata = utils.load_data('borcherding_test')
print(len(adata))

train, val = group_shuffle_split(adata, group_col='clonotype', val_split=0.2, random_seed=random_seed)

adata.obs['set'] = 'train'
adata.obs.loc[val.obs.index, 'set'] = 'val'


params_experiment = {
    'study_name': f'borcherding_moe_full',
    'comet_workspace': None,  # 'borcherding',
    'model_name': 'moe',
    'balanced_sampling': 'clonotype',
    'metadata': ['clonotype', 'Sample', 'Type', 'Tissue', 'functional.cluster'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna', 'borcherding_moe')
}

params_optimization = {
    'name': 'pseudo_metric',
    'prediction_labels': ['clonotype']#'functional.cluster']
}

timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)
