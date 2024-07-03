import mvtcr.evaluation.Metrics as Metrics
import scanpy as sc
import numpy as np

def run_knn_within_set_evaluation(data_full, embedding_function, prediction_labels, subset='val'):
    """
    Function for evaluating the embedding quality based upon kNN (k=1) prediction inside the set
    :param data_full: anndata object containing the full cell data (TCR + Genes) (train, val, test)
    :param embedding_function: function calculating the latent space for a single input
    :param prediction_labels: str or list[str], column name in data_full.obs that we want to predict
    :param subset: str to choose the set for testing
    :return: dictionary {metric: summary} containing the evaluation scores
    """
    if type(subset) == str:
        subset = [subset]
    if type(prediction_labels) == str:
        prediction_labels = [prediction_labels]
    data_test = data_full[data_full.obs['set'].isin(subset)]
    latent_tmp = embedding_function(data_test)
    scores = {}
    sc.pp.neighbors(latent_tmp, n_neighbors=2, knn=True)
    for prediction_label in prediction_labels:
        latent_tmp = latent_tmp[(latent_tmp.obs[prediction_label] != -99).to_numpy()]
        #IDEA latent tmp for f1 or calc just a knn graph and eval with that --> speed and mem efficient?
        scores[f'weighted_f1_{prediction_label}'] = Metrics.get_knn_f1_within_set(latent_tmp, prediction_label)
    return scores
