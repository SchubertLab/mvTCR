import optuna
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

import tcr_embedding.utils_training as utils_train

from tcr_embedding.config_optuna.mlp_count import suggest_params
from tcr_embedding.models.architectures.mlp_count_prediction import build_mlp
from tcr_embedding.models.losses.msle import MSLE


class DecisionHead(pl.LightningModule,):
    def __init__(self, params_model, n_in, n_out):
        super().__init__()
        self.save_hyperparameters()
        self.params_model = params_model
        self.network = build_mlp(params_model['mlp'], n_in, n_out).float()

        self.loss_function = MSLE()

    def forward(self, x):
        prediction = self.network(x)
        return prediction

    def training_step(self, batch, _):
        x, y_truth = batch
        x = x.float()
        y_truth = y_truth.float()
        y_pred = self.network(x)
        loss = self.loss_function(y_pred, y_truth)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y_truth = batch
        x = x.float()
        y_truth = y_truth.float()
        y_pred = self.network(x)
        loss = self.loss_function(y_pred, y_truth)
        self.log('val_loss', loss)
        # todo r2?
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params_model['learning_rate'])
        return optimizer


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float64)
        self.y = y.astype(np.float64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]


def embed_data(adata, path_model, key_counts):
    model = utils_train.load_model(adata, path_model)
    embedding = model.get_latent(adata, metadata=['set'], return_mean=True)
    embedding.obsm[key_counts] = adata.obsm[key_counts]
    return embedding


def get_training_data(embedding, key_counts):
    mask_train = embedding.obs['set'] == 'train'
    x_train = embedding.X[mask_train]
    y_train = embedding.obsm[key_counts][mask_train].toarray()

    mask_val = embedding.obs['set'] == 'val'
    x_val = embedding.X[mask_val]
    y_val = embedding.obsm[key_counts][mask_val].toarray()
    return CustomDataset(x_train, y_train), CustomDataset(x_val, y_val)


def objective(trial, data, params_experiment):
    path_save = os.path.join(params_experiment['save_path'], f'trial_{trial.number}')
    params_model = suggest_params(trial)

    train_data, val_data = get_training_data(data, params_experiment['key_prediction'])

    n_in = data.X.shape[1]
    n_out = data.obsm[params_experiment['key_prediction']].shape[1]

    decision_head = DecisionHead(params_model, n_in, n_out)
    train_loader = DataLoader(train_data, batch_size=params_model['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params_model['batch_size'], shuffle=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=path_save,
        filename='best_model',
        save_top_k=1,
        verbose=False,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = pl.callbacks.EarlyStopping('val_loss', patience=10)

    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], default_root_dir=path_save)
    trainer.fit(decision_head, train_loader, val_loader)
    return trainer.checkpoint_callback.best_model_score


def run_model_selection(adata, params_experiment, num_samples, timeout=None, n_jobs=1):
    sampler = optuna.samplers.TPESampler(seed=42)  # Make the sampler behave in a deterministic way.
    storage = f'sqlite:///{params_experiment["save_path"]}.db'
    if os.path.exists(params_experiment['save_path'] + '.db'):
        os.remove(params_experiment['save_path'] + '.db')

    study = optuna.create_study(study_name=params_experiment['study_name'], sampler=sampler, storage=storage,
                                direction='minimize', load_if_exists=False)

    embedded_data = embed_data(adata, params_experiment['model_path'], params_experiment['key_prediction'])

    study.optimize(lambda trial: objective(trial, embedded_data, params_experiment),
                   n_trials=num_samples, timeout=timeout, n_jobs=n_jobs)  # todo

    best_trial = study.best_trial
    print('Best trial: ')
    print(f'  trial_{best_trial.number}')
    print(f'  Value: {best_trial.value}')
