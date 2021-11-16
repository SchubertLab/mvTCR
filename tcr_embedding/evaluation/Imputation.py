import numpy as np
import random
import tcr_embedding.evaluation.WrapperFunctions as Wrapper
import tcr_embedding.evaluation.Metrics as Metrics

import scanpy as sc


def run_imputation_evaluation(data_full, embedding_function, query_source='val',
                              num_neighbors=5, label_pred='binding_name'):
    """
    Function for evaluating the embedding quality based upon imputation in the 10x dataset
    :param data_full: anndata object containing the full cell data (TCR + Genes) (train, val, test)
    :param embedding_function: function calculating the latent space for a single input
    :param query_source: str 'val' or 'test' to choose between evaluation mode
    :param num_neighbors: amount of neighbors for knn classification
    :param label_pred: label of the collumn used for prediction
    :return: dictionary {metric: summary} containing the evaluation scores
    """
    data_atlas = data_full[data_full.obs['set'] == 'train']
    data_query = data_full[data_full.obs['set'] == query_source]

    assert len(data_query) > 0, 'Empty query set. Specifier are "val" or "test"'

    embedding_atlas = embedding_function(data_atlas)
    embedding_query = embedding_function(data_query)

    scores = get_imputation_scores(embedding_atlas, embedding_query,
                                   data_atlas.obs[label_pred], data_query.obs[label_pred],
                                   num_neighbors=num_neighbors)
    return scores


def get_imputation_scores(embedding_atlas, embedding_query, label_atlas, label_embedding, num_neighbors=5):
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

    # Recall at k
    ks = [1, 10, 100]
    # recalls = Metrics.get_recall_at_k(embedding_atlas, embedding_query, label_atlas, label_embedding, ks)
    # summary['R@k'] = recalls

    # kNN score
    knn_score = Metrics.get_knn_classification(embedding_atlas, embedding_query, label_atlas, label_embedding,
                                           num_neighbors=num_neighbors, weights='distance')
    summary['knn'] = knn_score
    return summary


if __name__ == '__main__':
    """ For testing purposes """
    test_embedding_func = Wrapper.get_random_prediction_function(hidden_dim=800)
    print('Reading data')
    test_data = sc.read('../../data/10x_CD8TC/v5_train_val_test.h5ad')
    random.seed(29031995)
    # test_data = test_data[[random.randint(0, test_data.shape[0]-1) for _ in range(int(test_data.shape[0]*0.1))]]
    print('Start evaluation')
    res = run_imputation_evaluation(test_data, test_embedding_func, query_source='val',
                                    use_non_binder=True, use_reduced_binders=True)
    print(res)
