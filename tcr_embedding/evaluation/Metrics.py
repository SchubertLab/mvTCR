"""
File containing the metrics for evaluating the embedding space
"""

import numpy as np
from tqdm import tqdm
import time

from bottleneck import argpartition
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def recall_at_k(data_atlas, data_query, labels_atlas, labels_query, ks):
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


def knn_classification(data_atlas, data_query, labels_atlas, labels_query, num_neighbors=5, weights='distance'):
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
    report = classification_report(labels_query, labels_predicted)
    return report
