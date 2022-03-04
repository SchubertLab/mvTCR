import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import random

from tcr_embedding.dataloader.Dataset import JointDataset


def create_datasets(adata, val_split, metadata=None, conditional=None, labels=None, beta_only=False):
    """
    Create torch Dataset, see above for the input
    :param adata: list of adatas
    :param val_split:
    :param metadata:
    :param conditional:
    :param labels:
    :return: train_dataset, val_dataset, train_masks (for continuing training)
    """
    if metadata is None:
        metadata = []

    # Splits everything into train and val
    if val_split is not None:
        train_mask = (adata.obs[val_split] == 'train').values
    else:
        train_mask = np.ones(shape=(len(adata), ), dtype=bool)

    # Save dataset splits
    rna_train = adata.X[train_mask]
    rna_val = adata.X[~train_mask]

    if beta_only:
        tcr_seq = np.concatenate([adata.obsm['beta_seq']], axis=1)
        tcr_length = np.vstack([adata.obs['beta_len']]).T
    else:
        tcr_seq = np.concatenate([adata.obsm['alpha_seq'], adata.obsm['beta_seq']], axis=1)
        tcr_length = np.vstack([adata.obs['alpha_len'], adata.obs['beta_len']]).T
    tcr_train = tcr_seq[train_mask]
    tcr_val = tcr_seq[~train_mask]

    tcr_length_train = tcr_length[train_mask].tolist()
    tcr_length_val = tcr_length[~train_mask].tolist()

    metadata_train = adata.obs[metadata][train_mask].to_numpy()
    metadata_val = adata.obs[metadata][~train_mask].to_numpy()

    if conditional is not None:
        conditional_train = adata.obsm[conditional][train_mask]
        conditional_val = adata.obsm[conditional][~train_mask]
    else:
        conditional_train = None
        conditional_val = None

    train_dataset = JointDataset(rna_train, tcr_train, tcr_length_train, metadata_train,
                                 None, conditional_train)
    val_dataset = JointDataset(rna_val, tcr_val, tcr_length_val, metadata_val,
                               None, conditional_val)

    return train_dataset, val_dataset, train_mask


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# <- functions for the main data loader ->
def initialize_data_loader(adata, metadata, conditional, label_key, balanced_sampling, batch_size, beta_only=False):
    train_datasets, val_datasets, train_mask = create_datasets(adata, 'set', metadata, conditional, label_key,
                                                               beta_only=beta_only)

    if balanced_sampling is None:
        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
    else:
        sampling_weights = calculate_sampling_weights(adata, train_mask, class_column=balanced_sampling)
        sampler = WeightedRandomSampler(weights=sampling_weights, num_samples=len(sampling_weights),
                                        replacement=True)
        # shuffle is mutually exclusive to sampler, but sampler is anyway shuffled
        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=False,
                                  sampler=sampler, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def calculate_sampling_weights(adata, train_mask, class_column):
    """
    Calculate sampling weights for more balanced sampling in case of imbalanced classes,
    :params class_column: str, key for class to be balanced
    :params log_divisor: divide the label counts by this factor before taking the log, higher number makes the sampling more uniformly balanced
    :return: list of weights
    """
    label_counts = []

    label_count = adata[train_mask].obs[class_column].map(adata[train_mask].obs[class_column].value_counts())
    label_counts.append(label_count)

    label_counts = pd.concat(label_counts, ignore_index=True)
    label_counts = np.log(label_counts / 10 + 1)
    label_counts = 1 / label_counts

    sampling_weights = label_counts / sum(label_counts)
    return sampling_weights


# <- data loader for prediction ->
def initialize_prediction_loader(adata, metadata, batch_size, beta_only=False, conditional=None):
    prediction_dataset, _, _ = create_datasets(adata, val_split=None, conditional=conditional,
                                               metadata=metadata, beta_only=beta_only)
    prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)
    return prediction_loader


# <- data loader for calculating the transcriptome from the latent space ->
def initialize_latent_loader(adata_latent, batch_size, conditional):
    if conditional is None:
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(adata_latent.X))
    else:
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(adata_latent.X),
                                                 torch.from_numpy(adata_latent.obsm[conditional]))
    latent_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return latent_loader
