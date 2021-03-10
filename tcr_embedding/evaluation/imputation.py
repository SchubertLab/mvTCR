import numpy as np
import random
import tcr_embedding.evaluation.random_baseline as random_baseline
import tcr_embedding.evaluation.Metrics as Metrics

import scanpy as sc


def run_imputation_evaluation(data_atlas, data_query, embedding_function):
    """
    Function for evaluating the embedding quality based upon imputation in the 10x dataset
    :param data_atlas: anndata object containing the atlas cell data (TCR + Genes)
    :param data_query: anndata object containing the query cell data
    :param embedding_function: function calculating the latent space for a single input
    :return:
    """
    data_atlas = add_labels(data_atlas)
    data_query = add_labels(data_query)

    data_atlas, data_query = filter_data(data_atlas, data_query)

    embedding_atlas = embedding_function(data_atlas)
    embedding_query = embedding_function(data_query)

    scores = get_imputation_score(embedding_atlas, embedding_query,
                                  data_atlas.obs['binding_label'], data_query.obs['binding_label'])
    return scores


def filter_data(data_atlas, data_query):
    """
    Select data on which evaluation can be performed
    :param data_atlas: annData object containing the full atlas cell data
    :param data_query: anndata object containing the full query cell data
    :return: 2 anndata objects containing only the filtered data
    """
    # todo: filter multi chains
    # todo: filter non overlapping epitopes

    def general_filter(data):
        data = data[data.obs['has_ir'] == 'True']
        data = data[data.obs['multi_chain'] == 'False']
        return data
    print(len(data_atlas))
    data_atlas = general_filter(data_atlas)
    print(len(data_atlas))
    print('---')
    print(len(data_query))
    data_query = general_filter(data_query)
    print(len(data_query))

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
    ks = [1, 10, 100]
    recalls = Metrics.recall_at_k(embedding_atlas, embedding_query, label_atlas, label_embedding, ks)

    summary['R@k'] = recalls
    return summary


if __name__ == '__main__':
    test_embedding_func = random_baseline.random_embedding_function(hidden_dim=100)
    print('Reading data')
    test_data = sc.read('../../data/10x_CD8TC/highly_var_5000.h5ad')
    test_data_a = test_data[[random.randint(0, test_data.shape[0]-1) for _ in range(int(test_data.shape[0]*0.6))]]
    test_data_q = test_data[[random.randint(0, test_data.shape[0]-1) for _ in range(int(test_data.shape[0]*0.2))]]
    print('Start imputation')
    res = run_imputation_evaluation(test_data_a, test_data_q, test_embedding_func)
    print(res)