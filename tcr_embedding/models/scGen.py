import torch
import numpy as np
from anndata import AnnData

import tcr_embedding.utils_training as loading
import tcr_embedding.models.single_model as single
import tcr_embedding.evaluation.Pertubation as Eval


class PertubationPredictor:
    def __init__(self, model, data_source='bcc', verbosity=1):
        self.verbosity = verbosity

        self.print_verbose('-Loading Data')
        self.data = loading.load_data(data_source)

        self.print_verbose('-Loading Model')
        self.model = model

    @classmethod
    def from_model_checkpoint(cls, path_model, path_config, model_type, data_source='bcc', verbosity=1):
        """
        Initialize with path to model instead of model
        :param path_model:
        :param path_config:
        :param model_type:
        :param data_source:
        :param verbosity:
        :return: initialized object
        """
        model = loading.load_model(loading.load_data(data_source), path_config, path_model, model_type)
        return cls(model, data_source=data_source, verbosity=verbosity)

    def evaluate_pertubation(self, pertubation, splitting_criteria, per_column, indicator='pre'):
        """
        Evaluate the pertubation prediction
        :param pertubation: name of the column used to indicate peturbation state
        :param splitting_criteria: dict {column: value} how to split the dataset for validation
        :param per_column: None: calculate 1 delta on data, str: use individual delta per group in this column
        :param do_test: bool, if true the final test set is used
        :param indicator: str indicates what symbol in pertubation column represents pre and post
        :return: Score indicating the performance of the pertubation model
        """
        data_train, data_val = self.split_data(splitting_criteria)

        self.print_verbose('-Predicting Pertubation')
        prediction = self.predict_pertubation(data_train, data_val, per_column=per_column, indicator=indicator)

        self.print_verbose('-Evaluating Prediction')
        score = Eval.evaluate_pertubation(data_val, prediction, per_column, pertubation, indicator='pre')
        return score

    def split_data(self, splitting_criteria):
        """
        Split data to training and validation set.
        :param splitting_criteria: dictionary {adata.obs_column: value} which samples to select to validation set
        :param do_test: Whether or not to use the Training set or Validation set
        :return: 2 adata objects with training und validation data
        """
        masks = []  # todo change set_pertubation
        for key, value in splitting_criteria.items():
            mask_cur = self.data.obs[key] == value
            masks.append(mask_cur.values)
        mask = np.stack(masks)
        mask = np.all(mask, axis=0)
        data_val = self.data[mask]
        data_train = self.data[~mask]
        return data_train, data_val

    def predict_pertubation(self, data_train, data_val, per_column, indicator='pre'):
        """
        Calculate the perturbed transcriptome data.
        :param data_train: adata object containing the training data
        :param data_val: adata object containing the validation data
        :param per_column: None: calculate 1 delta on data, str: use individual delta per group in this column
        :param indicator: str indicates what symbol in pertubation column represents pre and post
        :return: adata object containing predicted transcriptome data
        """
        latent_train, latent_val = self.calculate_latent_space(data_train, data_val)
        self.print_verbose('--Predict Validation Post Latent Space')
        latent_prediction = self.predict_latent(latent_train, latent_val, pertubation='treatment',
                                                per_column=per_column, indicator=indicator)
        self.print_verbose('--Predict Transcriptome Post')
        cell_prediction = self.predict_cells(latent_prediction)
        return cell_prediction

    def calculate_latent_space(self, data_train, data_val):
        """
        Calculates the latent space of the VAE
        :param data_train: adata object with training data
        :param data_val: adata object with validation data
        :return: 2 adata objects containing the latent spaces for training and validation set
        """
        metadata = ['clonotype', 'cluster', 'cluster_tcr', 'treatment', 'patient']
        self.print_verbose('--Calculating Training Latent Spaces')
        latent_train = self.model.get_latent([data_train], metadata=metadata, batch_size=1024, return_mean=True,
                                             device='cuda')
        self.print_verbose('--Calculating Validation Latent Spaces')
        latent_val = self.model.get_latent([data_val], metadata=metadata, batch_size=1024, return_mean=True,
                                           device='cuda')
        return latent_train, latent_val

    def predict_latent(self, latent_train, latent_val, pertubation, indicator='pre', per_column=None):
        """
        Predict the latent space of cells post pertubation
        :param latent_train: adata object with training latent space
        :param latent_val: adata object with validation latent space
        :param pertubation: column name in adata.obs indictating the pertubation state
        :param indicator: value in pertubation column used to indicate pre state
        :param per_column: calculate delta for each value of this column, if None over all values
        :return: adata object containing the predicted latent space
        """
        train_pre = latent_train[latent_train.obs[pertubation] == indicator]
        train_post = latent_train[latent_train.obs[pertubation] != indicator]
        val_pre = latent_val[latent_val.obs[pertubation] == indicator]

        if per_column is not None:
            self.print_verbose('--Calculate deltas')
            deltas = {}
            for group in latent_train.obs[per_column].unique():
                avg_pre = train_pre[train_pre.obs[per_column] == group]
                avg_pre = np.mean(avg_pre.X, axis=0)
                avg_post = train_post[train_post.obs[per_column] == group]
                avg_post = np.mean(avg_post.X, axis=0)
                deltas[group] = avg_post - avg_pre

            self.print_verbose('--Apply deltas')
            prediction = val_pre.copy()

            for group, delta in deltas.items():
                prediction.X[prediction.obs[per_column] == group] += delta
        else:
            # todo introduce cell type balancing
            self.print_verbose('--Calculate deltas')
            avg_pre = np.mean(train_pre.X, axis=0)
            avg_post = np.mean(train_post.X, axis=0)
            delta = avg_post - avg_pre
            self.print_verbose('--Apply deltas')
            prediction = val_pre.copy()
            prediction.X += delta
        return prediction

    def predict_cells(self, latent_space):
        """
        Predict transcriptome from latent space representation
        :param latent_space: latent space representation of the cells
        :return: adata object containing the predicted transcriptome data
        """
        data_in = torch.Tensor(latent_space.X).to('cuda:0')
        joint_encoding = self.model.model.shared_decoder(data_in)
        prediction_cells = self.model.model.gene_decoder(joint_encoding).detach().cpu().numpy()
        predicted_adata = AnnData(
            X=prediction_cells,
            obs=latent_space.obs.copy(),
            var=self.data.var.copy(),
            obsm=latent_space.obsm.copy(),
        )
        return predicted_adata

    def print_verbose(self, statement):
        """
        Print function with builtin verbosity.
        :param statement: String to be printed
        :return: prints to console
        """
        if self.verbosity == 0:
            return
        if self.verbosity == 1:
            print(statement)


if __name__ == '__main__':
    pertubator = PertubationPredictor.from_model_checkpoint(
                                        path_model='saved_models/bcc_selected/singleRNA/20210505_05-45-59_67f8291e/'
                                                 'checkpoint_0/bcc_tune_single_bcc_scRNA_best_rec_model.pt',
                                        path_config='saved_models/bcc_selected/singleRNA/20210505_05-45-59_67f8291e/',
                                        model_type=single.SingleModel,
                                        data_source='bcc', verbosity=1)
    pertubator.evaluate_pertubation(pertubation='treatment', splitting_criteria={'patient': 'su009'},
                                    per_column='cluster', indicator='pre')
