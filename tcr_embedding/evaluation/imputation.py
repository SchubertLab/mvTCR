import numpy as np
import random
import tcr_embedding.evaluation.WrapperFunctions as Wrapper
import tcr_embedding.evaluation.Metrics as Metrics

import scanpy as sc


def run_imputation_evaluation(data_atlas, data_query, embedding_function, use_non_binder=True):
    """
    Function for evaluating the embedding quality based upon imputation in the 10x dataset
    :param data_atlas: anndata object containing the atlas cell data (TCR + Genes)
    :param data_query: anndata object containing the query cell data
    :param embedding_function: function calculating the latent space for a single input
    :param use_non_binder: bool filter out non binding TCRs
    :return: dictionary {metric: summary} containing the evaluation scores
    """
    data_atlas = add_labels(data_atlas)
    data_query = add_labels(data_query)

    data_atlas, data_query = filter_data(data_atlas, data_query, use_non_binder=use_non_binder)

    embedding_atlas = embedding_function(data_atlas)
    embedding_query = embedding_function(data_query)

    scores = get_imputation_score(embedding_atlas, embedding_query,
                                  data_atlas.obs['binding_label'], data_query.obs['binding_label'])
    return scores


def filter_data(data_atlas, data_query, use_non_binder=True):
    """
    Select data on which evaluation can be performed
    :param data_atlas: annData object containing the full atlas cell data
    :param data_query: anndata object containing the full query cell data
    :return: 2 anndata objects containing only the filtered data
    """
    def general_filter(data):
        """
        Filter that should be applied to both datasets
        :param data: anndata object containing cell data
        :return: filtered anndata object
        """
        data = data[data.obs['has_ir'] == 'True']
        data = data[data.obs['multi_chain'] == 'False']
        if not use_non_binder:
            data = data[data.obs['binding_label'] != 'non_binder']
        return data

    data_atlas = general_filter(data_atlas)
    data_query = general_filter(data_query)

    return data_atlas, data_query


def add_labels(data):
    """
    Extracts list of epitope specificity from cell data
    :param data: anndata object containing the cell data
    :return: list (num cells) containing the binder
    """
    labels = []
    for idx, row in data.obs.iterrows():
        label_row = 'non_binder'
        for col in [col for col in data.obs if col.endswith('binder')]:
            if row[col] == 'True':
                label_row = col
        labels.append(label_row)
    data.obs['binding_label'] = labels
    return data


def get_imputation_score(embedding_atlas, embedding_query, label_atlas, label_embedding):
    """
    Calculate evualation scores based on imputation, so far Recall@k
    :param embedding_atlas: numpy array (num cells, hidden dim) of the atlas dataset
    :param embedding_query: see above of the query dataset
    :param label_atlas: epitope specificity by cell of the atlas
    :param label_embedding: epitope specificity by cell of the query set
    :return: dictionary {metric_name: metric_score} containing R@1, R@10, R@100, ...
    """
    summary = {}

    # Recall at k
    ks = [1, 10, 100]
    recalls = Metrics.recall_at_k(embedding_atlas, embedding_query, label_atlas, label_embedding, ks)
    summary['R@k'] = recalls

    # kNN score
    knn_score = Metrics.knn_classification(embedding_atlas, embedding_query, label_atlas, label_embedding,
                                           num_neighbors=5, weights='distance')
    summary['knn'] = knn_score
    return summary


if __name__ == '__main__':
    """ For testing purposes: """
    test_embedding_func = Wrapper.get_random_prediction_function(hidden_dim=100)
    print('Reading data')
    test_data = sc.read('../../data/10x_CD8TC/highly_var_5000.h5ad')
    test_data_a = test_data[[random.randint(0, test_data.shape[0]-1) for _ in range(int(test_data.shape[0]*0.05))]]
    test_data_q = test_data[[random.randint(0, test_data.shape[0]-1) for _ in range(int(test_data.shape[0]*0.05))]]
    print('Start evaluation')
    res = run_imputation_evaluation(test_data_a, test_data_q, test_embedding_func, use_non_binder=False)
    print(res)
