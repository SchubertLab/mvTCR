from tcr_embedding.evaluation.PertubationPrediction import evaluate_pertubation
import numpy as np
import scanpy as sc
import time
from sklearn.neighbors import NearestNeighbors

sc.settings.verbosity = 0


def predict_pertubation(latent_train, latent_val, model, column_perturbation, indicator_perturbation, var_names,
                        col_type, return_latent=False):
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
    latent_val_pre = latent_val[latent_val.obs[column_perturbation] == indicator_perturbation]

    cts_pre = latent_train[latent_train.obs[column_perturbation] == indicator_perturbation].obs[col_type].unique()
    cts_post = latent_train[latent_train.obs[column_perturbation] != indicator_perturbation].obs[col_type].unique()
    cts_both = [ct for ct in cts_pre if ct in cts_post]

    centers = latent_train[latent_train.obs[col_type].isin(cts_both) &
                           (latent_train.obs[column_perturbation] == indicator_perturbation)]
    centers = [centers[centers.obs[col_type] == ct_group].X.mean(axis=0) for ct_group in cts_both]

    center_pre = latent_val.X.mean(axis=0)

    nn = NearestNeighbors(1, metric='euclidean', algorithm='brute').fit(centers)
    _, idx = nn.kneighbors(center_pre.reshape(1, -1))
    idx = idx[0][0]

    center_post = latent_train[(latent_train.obs[col_type] == cts_both[idx]) &
                               (latent_train.obs[column_perturbation] != indicator_perturbation)].X.mean(axis=0)
    delta = center_post - centers[idx]

    ad_pred = sc.AnnData(latent_val_pre.X + delta, obs=latent_val.obs.copy())
    if return_latent:
        return ad_pred

    ad_pred = model.predict_rna_from_latent(ad_pred, metadata=ad_pred.obs.columns)
    ad_pred.var_names = var_names
    return ad_pred


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


def run_scgen_cross_validation(adata, column_fold, model, column_perturbation, indicator_perturbation, degs=None):
    """
    Runs perturbation prediction over a specified fold column and evaluates the results
    :param adata: adata object, of the raw data
    :param column_fold: str, indicating the column over which to cross validate
    :param model: model used for latent space generation
    :param column_perturbation: str, column in the adata object indicating the pertubation
    :param indicator_perturbation: str, value for the 'pre' state of the profile after perturbation
    :return: dict, summary over performance on the different splits and aggregation
    """
    tic = time.time()
    latent_full = model.get_latent(adata, metadata=[column_fold, column_perturbation])

    summary_performance = {}
    rs_squared = []

    cats_pre = adata[adata.obs[column_perturbation] == indicator_perturbation].obs[column_fold].value_counts()
    cats_pre = [cat for cat, count in cats_pre.items() if count > 5]
    cats_post = adata[adata.obs[column_perturbation] != indicator_perturbation].obs[column_fold].value_counts()
    cats_post = [cat for cat, count in cats_post.items() if count > 5]
    cats_both = [cat for cat in cats_pre if cat in cats_post]

    centers_pre = [latent_full[(latent_full.obs[column_fold] == cat_group) &
                               (latent_full.obs[column_perturbation] == indicator_perturbation)].X.mean(axis=0)
                   for cat_group in cats_both]
    centers_post = [latent_full[(latent_full.obs[column_fold] == cat_group) &
                                (latent_full.obs[column_perturbation] != indicator_perturbation)].X.mean(axis=0)
                    for cat_group in cats_both]

    for i, fold in enumerate(cats_both):
        cats_tmp = cats_both.copy()
        cats_tmp.pop(i)
        centers_pre_tmp = centers_pre.copy()
        centers_pre_tmp.pop(i)
        centers_post_tmp = centers_post.copy()
        centers_post_tmp.pop(i)

        mask_train = latent_full.obs[column_fold] != fold
        latent_val = latent_full[~mask_train]
        latent_val_pre = latent_val[latent_val.obs[column_perturbation] == indicator_perturbation]

        center_pre = latent_val_pre.X.mean(axis=0)
        nn = NearestNeighbors(1, metric='euclidean', algorithm='brute').fit(centers_pre_tmp)
        _, idx = nn.kneighbors(center_pre.reshape(1, -1))
        idx = idx[0][0]

        delta = centers_post_tmp[idx] - centers_pre_tmp[idx]

        ad_pred = sc.AnnData(latent_val_pre.X + delta, obs=latent_val_pre.obs.copy())
        ad_pred = model.predict_rna_from_latent(ad_pred, metadata=ad_pred.obs.columns)
        ad_pred.var_names = adata.var_names

        data_ct = adata[adata.obs[column_fold] == fold].copy()
        score = evaluate_pertubation(data_ct.copy(), ad_pred, column_fold, column_perturbation,
                                     indicator=indicator_perturbation, gene_set=degs)

        for key, value in score.items():
            summary_performance[f'{fold}_key'] = value
        rs_squared.append(score['degs']['r_squared'])

    summary_performance['avg_r_squared'] = sum(rs_squared) / len(rs_squared)
    toc = time.time()
    #print(summary_performance['avg_r_squared'])
    #print(toc - tic)
    return summary_performance
