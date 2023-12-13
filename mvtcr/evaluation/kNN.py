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
    for prediction_label in prediction_labels:
        #TODO
        #print('knn 1', prediction_label)
        #print('knn 2', latent_tmp.obsm[prediction_label].isnull().unique())
        #print('knn 3', [latent_tmp.obsm[prediction_label] != 'nan'])        
        #print('knn 5', [latent_tmp.obsm[prediction_label] != -99])
        #print('knn 1', (latent_tmp.obsm[prediction_label] != 'nan').to_numpy())
        

        '''latent_tmp = latent_tmp[(latent_tmp.obsm[prediction_label] != 'nan').to_numpy()]


        latent_tmp = latent_tmp[(latent_tmp.obsm[prediction_label] != -99).to_numpy()]
        latent_tmp = latent_tmp[~latent_tmp.obsm[prediction_label].isnull()]'''
        #TODO n_neighbors 2? magic number?
        sc.pp.neighbors(latent_tmp, n_neighbors=2, knn=True)
        scores[f'weighted_f1_{prediction_label}'] = Metrics.get_knn_f1_within_set(latent_tmp, prediction_label)
    return scores
