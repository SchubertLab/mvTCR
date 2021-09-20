import tcr_embedding.evaluation.Metrics as Metrics
import scanpy as sc


def run_knn_within_set_evaluation(data_full, embedding_function, prediction_labels, set='val'):
    """
    Function for evaluating the embedding quality based upon kNN (k=1) prediction inside the set
    :param data_full: anndata object containing the full cell data (TCR + Genes) (train, val, test)
    :param embedding_function: function calculating the latent space for a single input
    :param prediction_labels: str or list[str], column name in data_full.obs that we want to predict
    :param set: str to choose the set for testing
    :return: dictionary {metric: summary} containing the evaluation scores
    """
    if type(prediction_labels) == str:
        prediction_labels = [prediction_labels]
    data_test = data_full[data_full.obs['set'] == set]
    latent_test = embedding_function(data_test)
    sc.pp.neighbors(latent_test, n_neighbors=2, knn=True)
    scores = {}
    for prediction_label in prediction_labels:
        scores[f'weighted_f1_{prediction_label}'] = Metrics.get_knn_f1_within_set(latent_test, prediction_label)
    return scores
