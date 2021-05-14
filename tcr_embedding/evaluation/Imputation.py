import numpy as np
import random
import tcr_embedding.evaluation.WrapperFunctions as Wrapper
import tcr_embedding.evaluation.Metrics as Metrics

import scanpy as sc


def run_imputation_evaluation(data_full, embedding_function, query_source='val', use_non_binder=True,
                              use_reduced_binders=True, num_neighbors=5):
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

    data_atlas, data_query = filter_data(data_atlas, data_query, use_non_binder=use_non_binder,
                                         use_reduced_binders=use_reduced_binders)

    embedding_atlas = embedding_function(data_atlas)
    embedding_query = embedding_function(data_query)

    scores = get_imputation_scores(embedding_atlas, embedding_query,
                                   data_atlas.obs['binding_name'], data_query.obs['binding_name'],
                                   num_neighbors=num_neighbors)
    return scores


def filter_data(data_atlas, data_query, use_non_binder=True, use_reduced_binders=True):
    """
    Select data on which evaluation is performed
    :param data_atlas: annData object containing the full atlas cell data
    :param data_query: anndata object containing the full query cell data
    :param use_non_binder: if true tcrs without specificity are filtered out
    :param use_reduced_binders: if true antigen with low amount of tcrs are regarded as non binders
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
        if use_reduced_binders:
            # List of antigens from David Fischer's paper, basically the 8 most common antigens
            high_antigen_count = ['A0201_ELAGIGILTV_MART-1_Cancer_binder',
                                  'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
                                  'A0201_GLCTLVAML_BMLF1_EBV_binder',
                                  'A0301_KLGGALQAK_IE-1_CMV_binder',
                                  'A0301_RLRAEAQVK_EMNA-3A_EBV_binder',
                                  'A1101_IVTDFSVIK_EBNA-3B_EBV_binder',
                                  'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder',
                                  'B0801_RAKFKQLL_BZLF1_EBV_binder']
            data.obs['binding_label'][~data.obs['binding_name'].isin(high_antigen_count)] = -1
            data.obs['binding_name'][~data.obs['binding_name'].isin(high_antigen_count)] = 'no_data'
        if not use_non_binder:
            data = data[data.obs['binding_label'] != -1]
        return data

    data_atlas = general_filter(data_atlas)
    data_query = general_filter(data_query)
    return data_atlas, data_query


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
