from tcr_embedding.evaluation.PertubationPrediction import evaluate_pertubation
import numpy as np
import scanpy as sc
import time

sc.settings.verbosity = 0


def predict_pertubation(latent_train, latent_val, model, column_perturbation, indicator_perturbation, var_names,
                        return_latent=False, per_type=False, col_type=None):
    """
    Predict the effect of pertubation on transcriptome level
    :param latent_train: adata object containing the latent spaces of the training dataset
    :param latent_val: adata object containing the latent spaces of the validaiton dataset
    :param model: model to predict transcritome from latent space
    :param column_perturbation: str, column in the adata objects indicating the pertubation
    :param indicator_perturbation: str, value for the 'pre' state of the perturbation
    :param var_names: list containing the gene names
    :return: adata object, containing the predicted transcriptome profile after perturbation
    """
    # todo delta per cell type? # upsampling?
    if not per_type:
        delta = get_delta(latent_train, column_perturbation, indicator_perturbation)
        if col_type is None:
            pass
        else:
            deltas = []

            cats_pre = latent_train[latent_train.obs[column_perturbation] == indicator_perturbation].obs[
                col_type].value_counts()
            cats_pre = [cat for cat, count in cats_pre.items() if count > 0]
            cats_post = latent_train[latent_train.obs[column_perturbation] != indicator_perturbation].obs[
                col_type].value_counts()
            cats_post = [cat for cat, count in cats_post.items() if count > 0]
            cats_both = [cat for cat in cats_pre if cat in cats_post]

            for t in cats_both:
                d = get_delta(latent_train[latent_train.obs[col_type]==t], column_perturbation, indicator_perturbation)
                deltas.append(d)
            delta = np.vstack(deltas).mean(axis=0)
        adata_pred = sc.AnnData(latent_val.X + delta, obs=latent_val.obs.copy())
    else:
        adatas_pred = []
        for t in latent_train.obs[per_type].unique():
            delta = get_delta(latent_train[latent_train.obs[per_type] == t],
                              column_perturbation, indicator_perturbation)
            adata_tmp = latent_val[latent_val.obs[per_type] == t].copy()
            adata_new = sc.AnnData(adata_tmp.X + delta, obs=adata_tmp.obs)
            adatas_pred.append(adata_new)
        adata_pred = sc.concat(adatas_pred)
    if return_latent:
        return adata_pred
    adata_pred = model.predict_rna_from_latent(adata_pred, metadata=adata_pred.obs.columns)
    adata_pred.var_names = var_names
    return adata_pred


def get_delta(adata_latent, column_perturbation, indicator_perturbation):
    """
    Calculate the difference vector between pre and post perturbation in the latent space
    :param adata_latent: adata oject, containing the latent space representation of the training data
    :param column_perturbation: str, column in the adata object indicating the pertubation
    :param indicator_perturbation: str, value for the 'pre' state of the profile after perturbation
    :return: numpy array, difference vectore
    """
    mask_pre = adata_latent.obs[column_perturbation] == indicator_perturbation
    latent_pre = adata_latent[mask_pre]
    latent_post = adata_latent[~mask_pre]
    avg_pre = np.mean(latent_pre.X, axis=0)
    avg_post = np.mean(latent_post.X, axis=0)
    delta = avg_post - avg_pre
    return delta


def run_scgen_cross_validation(adata, column_fold, model, column_perturbation, indicator_perturbation):
    """
    Runs perturbation prediction over a specified fold column and evaluates the results
    :param adata: adata object, of the raw data
    :param column_fold: str, indicating the column over which to cross validate
    :param model: model used for latent space generation
    :param column_perturbation: str, column in the adata object indicating the pertubation
    :param indicator_perturbation: str, value for the 'pre' state of the profile after perturbation
    :return: dict, summary over performance on the different splits and aggregation
    """
    latent_full = model.get_latent(adata, metadata=[column_fold, column_perturbation])

    summary_performance = {}
    rs_squared = []

    cats_pre = adata[adata.obs[column_perturbation] == indicator_perturbation].obs[column_fold].value_counts()
    cats_pre = [cat for cat, count in cats_pre.items() if count > 10]
    cats_post = adata[adata.obs[column_perturbation] != indicator_perturbation].obs[column_fold].value_counts()
    cats_post = [cat for cat, count in cats_post.items() if count > 10]
    cats_both = [cat for cat in cats_pre if cat in cats_post]

    for fold in cats_both:
        mask_train = latent_full.obs[column_fold] != fold
        latent_train = latent_full[mask_train]
        latent_val = latent_full[~mask_train]

        latent_val_pre = latent_val[latent_val.obs[column_perturbation] == indicator_perturbation]
        pred_val_post = predict_pertubation(latent_train, latent_val_pre, model,
                                            column_perturbation, indicator_perturbation,
                                            var_names=adata.var_names, col_type=column_fold)

        score = evaluate_pertubation(adata[adata.obs[column_fold] == fold].copy(), pred_val_post, None,
                                     column_perturbation, indicator=indicator_perturbation)

        for key, value in score.items():
            summary_performance[f'{fold}_key'] = value
        rs_squared.append(score['all_genes']['r_squared'])

    summary_performance['avg_r_squared'] = sum(rs_squared) / len(rs_squared)
    return summary_performance
