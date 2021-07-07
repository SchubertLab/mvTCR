import numpy as np
import random
import tcr_embedding.evaluation.WrapperFunctions as Wrapper
import tcr_embedding.evaluation.Metrics as Metrics

import scanpy as sc


def run_knn_evaluation(data_full, embedding_function, name_label, query_source='val', num_neighbors=5):
    """
    Function for evaluating the embedding quality based upon imputation in the 10x dataset
    :param data_full: anndata object containing the full cell data (TCR + Genes) (train, val, test)
    :param embedding_function: function calculating the latent space for a single input
    :param query_source: str 'val' or 'test' to choose between evaluation mode
    :param use_non_binder: bool filter out non binding TCRs
    :param use_reduced_binders: if true antigen with low amount of tcrs are regarded as non binders
    :param num_neighbors: amount of neighbors for knn classification
    :return: dictionary {metric: summary} containing the evaluation scores
    """
    data_atlas = data_full[data_full.obs['set'] == 'train']
    data_query = data_full[data_full.obs['set'] == query_source]

    assert len(data_query) > 0, 'Empty query set. Specifier are "val" or "test"'

    embedding_atlas = embedding_function(data_atlas)
    embedding_query = embedding_function(data_query)

    scores = get_knn_scores(embedding_atlas, embedding_query, data_atlas.obs[name_label], data_query.obs[name_label], num_neighbors=num_neighbors)

    return scores


def get_knn_scores(embedding_atlas, embedding_query, label_atlas, label_embedding, num_neighbors=5):
    """
    Calculate evaluation scores based on imputation, so far knn classification report and Recall@k
    :param embedding_atlas: numpy array (num cells, hidden dim) of the atlas dataset
    :param embedding_query: see above of the query dataset
    :param label_atlas: epitope specificity by cell of the atlas
    :param label_embedding: epitope specificity by cell of the query set
    :param num_neighbors: amount of neighbors used for knn classification
    :return: dictionary {metric_name: metric_score}
    """
    summary = {}

    # kNN score
    knn_score = Metrics.get_knn_classification(embedding_atlas, embedding_query, label_atlas, label_embedding,
                                               num_neighbors=num_neighbors, weights='distance')
    summary['knn'] = knn_score
    return summary
