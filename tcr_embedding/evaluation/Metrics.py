"""
File containing the metrics for evaluating the embedding space
"""
import numpy as np
from tqdm import tqdm

from bottleneck import argpartition
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, silhouette_score, adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from scipy import stats


def get_recall_at_k(data_atlas, data_query, labels_atlas, labels_query, ks):
    """
    Calculates the Recall@k value: % in the k nearest neighbors, is there the correct label
    :param data_atlas: numpy array (num_cells, hidden_size) embeddings of the atlas data
    :param data_query: numpy array (num_cells, hidden_size) embeddings of the query data
    :param labels_atlas: list (num_cells) labels of the atlas data
    :param labels_query: list (num_cells) labels of the query data
    :param ks: list or int of k values indicating the num of nearest neighbors
    :return: dictonary {k: r@k value}
    """
    if isinstance(ks, int):
        ks = [ks]
    count_correct = defaultdict(lambda: 0)
    for i in tqdm(range(data_query.shape[0])):
        difference = data_atlas - data_query[i]
        distances = np.linalg.norm(difference, axis=-1)

        for k in ks:
            nns = argpartition(distances, k)[:k]
            if any(labels_query[i] == labels_atlas[nn] for nn in nns):
                count_correct[k] += 1
    for key, value in count_correct.items():
        count_correct[key] = value / data_query.shape[0]
    return dict(count_correct)


def get_knn_classification(data_atlas, data_query, labels_atlas, labels_query, num_neighbors=5, weights='distance'):
    """
    Evaluates with kNN based on scikit-learn
    :param data_atlas: numpy array (num_cells, hidden_size) embeddings of the atlas data
    :param data_query: numpy array (num_cells, hidden_size) embeddings of the query data
    :param labels_atlas: list (num_cells) labels of the atlas data
    :param labels_query: list (num_cells) labels of the query data
    :param num_neighbors: amount of neighbors used for kNN
    :param weights: kNN weighting,
    :return:
    """

    clf = KNeighborsClassifier(num_neighbors, weights)
    clf.fit(data_atlas, labels_atlas)

    labels_predicted = clf.predict(data_query)
    report = classification_report(labels_query, labels_predicted, output_dict=True)
    return report


def get_silhouette_scores(embeddings, labels_predicted):
    """
    Calculates the Silhouette score as internal cluster evaluation
    :param embeddings: numpy array (n_samples, dim_hidden) containing the latent embedding
    :param labels_predicted: predicted labels based on clustering in latent space
    :return:
    """
    try:
        score = silhouette_score(embeddings, labels_predicted, metric='euclidean', random_state=29031995)
    # If the number of cluster is 1, then ASW can't be calculated and raises an Error
    except:
        score = -99
    return score


def get_adjusted_mutual_information(labels_true, labels_predicted):
    """
    Calculates the AMI score as external cluster evaluation
    :param labels_true: ground truth labels for external cluster evaluation
    :param labels_predicted: predicted labels based on clustering in latent space
    :return: Adjusted mutual information score
    """
    scores = adjusted_mutual_info_score(labels_true, labels_predicted)
    return scores


def get_normalized_mutual_information(labels_true, labels_predicted):
    """
    Calculates the NMI score as external cluster evaluation
    :param labels_true: ground truth labels for external cluster evaluation
    :param labels_predicted: predicted labels based on clustering in latent space
    :return: Normalized mutual information score
    """
    scores = normalized_mutual_info_score(labels_true, labels_predicted, average_method='arithmetic')
    return scores


def get_adjusted_random_score(labels_true, labels_predicted):
    """
    Calculates the AMI score as external cluster evaluation
    :param labels_true: ground truth labels for external cluster evaluation
    :param labels_predicted: predicted labels based on clustering in latent space
    :return: Adjusted mutual information score
    """
    scores = adjusted_rand_score(labels_true, labels_predicted)
    return scores


def get_square_pearson(ground_truth, prediction):
    try:
        x = np.average(ground_truth.X.A, axis=0)
    except AttributeError:
        x = np.average(ground_truth.X, axis=0)
    try:
        y = np.average(prediction.X.A, axis=0)
    except AttributeError:
        y = np.average(prediction.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value**2
    report = {
        'm': m,
        'b': b,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'r_squared': r_squared
    }
    return report


def get_knn_f1_within_set(latent, column_name):
    con = latent.obsp['connectivities'].A.astype(np.bool)
    nearest_neighbor_label = [latent.obs[column_name].values[row].tolist()[0] for row in con]
    result = classification_report(latent.obs[column_name], nearest_neighbor_label, output_dict=True)
    result = result['weighted avg']['f1-score']
    return result
