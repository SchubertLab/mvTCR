"""
python -u borcherding.py --gpus 1
"""
# comet-ml must be imported before torch and sklearn
import comet_ml

import sys
sys.path.append('..')

import numpy as np

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


adata = utils.load_data('borcherding')
print(len(adata))


# Randomly select patients to be left out during training
def get_n_patients(amount_patients):
    if amount_patients <= 5:
        return 0
    else:
        return 2


holdout_patients = {}

adata.obs['Tissue+Type'] = [f'{tissue}.{type_}' for tissue, type_ in zip(adata.obs['Tissue'], adata.obs['Type'])]
counts = adata.obs.groupby('Tissue+Type')['Sample'].value_counts()
for cat in adata.obs['Tissue+Type'].unique():
    n = get_n_patients(len(counts[cat]))
    choice = np.random.choice(counts[cat].index, n, replace=False).tolist()
    holdout_patients[cat] = choice

for patients in holdout_patients.values():
    adata = adata[~adata.obs['Sample'].isin(patients)]
print(len(adata))


train, val = group_shuffle_split(adata, group_col='clonotype', val_split=0.2, random_seed=random_seed)
print(len(train))
print(len(val))
raise ValueError
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
