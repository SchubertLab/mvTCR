import random
import tcr_embedding.evaluation.WrapperFunctions as Wrapper
import tcr_embedding.evaluation.Metrics as Metrics

import scanpy as sc
from anndata import AnnData


def run_clustering_evaluation(data_full, embedding_function, source_data='val', name_label='clonotype',
                              cluster_params=None):
    """
    Function for evaluating the embedding quality based upon imputation in the 10x dataset
    :param data_full: anndata object containing the full cell data (TCR + Genes) (train, val, test)
    :param embedding_function: function calculating the latent space for a single input
    :param source_data: str 'val' or 'test' to choose between evaluation mode
    :param name_label: column in data_full.obs used as ground truth for external cluster evaluation
    :param cluster_params: parameters for leiden clustering, None for default
    :return: dictionary {metric: summary} containing the evaluation scores
    """
    data_eval = data_full[data_full.obs['set'] == source_data]
    data_eval = filter_data(data_eval)

    assert len(data_eval) > 0, 'Empty data set. Specifier are "val" or "test"'

    embeddings = embedding_function(data_eval)
    labels_true = data_eval.obs[name_label].to_numpy()
    labels_predicted = predict_clustering(embeddings, cluster_params)

    scores = get_clustering_scores(embeddings, labels_true, labels_predicted)
    return scores


def filter_data(data):
    """
    Select data on which evaluation is performed
    :param data: annData object containing the cell data
    :return: 2 anndata objects containing only the filtered data
    """
    data = data[data.obs['has_ir'] == 'True']
    data = data[data.obs['multi_chain'] == 'False']
    return data


def predict_clustering(embeddings, params):
    """
    Calculate cluster labels based on leiden algorithm
    :param embeddings: numpy array (num_samples, dim_hidden) representing the latent space embedding of each cell
    :param params: parameter of leiden clustering
    :return: numpy array (num_cells,) containing cluster annotation
    """
    if params is None:
        params = {'resolution': 1,
                  'num_neighbors': 5}
    adata = AnnData(embeddings)
    sc.pp.neighbors(adata, n_neighbors=params['num_neighbors'], use_rep='X', random_state=29031995)
    sc.tl.leiden(adata, resolution=params['resolution'], random_state=29031995)
    return adata.obs['leiden'].to_numpy()


def get_clustering_scores(embeddings, labels_true, labels_predicted):
    """
    Calculate evaluation scores based on clustering.
    Silhouette score for internal, Adjusted Mutual Information as external evaluation.
    :param embeddings: numpy array (num cells, hidden dim) of the cell dataset
    :param labels_true: ground truth cluster annotation for external evaluation
    :param labels_predicted: predicted cluster annotation for internal / external evaluation
    :return: dictionary {metric_name: metric_score}
    """
    summary = {
        'ASW': Metrics.get_silhouette_scores(embeddings, labels_predicted),
        'AMI': Metrics.get_adjusted_mutual_information(labels_true, labels_predicted),
        'NMI': Metrics.get_normalized_mutual_information(labels_true, labels_predicted),
        'ARI': Metrics.get_adjusted_rand_index(labels_true, labels_predicted)
    }
    return summary


if __name__ == '__main__':
    """ For testing purposes """

    test_embedding_func = Wrapper.get_random_prediction_function(hidden_dim=800)
    print('Reading data')
    test_data = sc.read('../../data/10x_CD8TC/v5_train_val_test.h5ad')
    random.seed(29031995)
    # test_data = test_data[[random.randint(0, test_data.shape[0]-1) for _ in range(int(test_data.shape[0]*0.1))]]
    print('Start Cluster evaluation')
    res = run_clustering_evaluation(test_data, test_embedding_func, source_data='val', name_label='clonotype')
    print(res)
