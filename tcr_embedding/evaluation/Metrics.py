"""
File containing the metrics for evaluating the embedding space
"""

import numpy as np
from tqdm import tqdm
import time
from bottleneck import argpartition
from collections import defaultdict


def recall_at_k(data_atlas, data_query, labels_atlas, labels_query, ks):
    """

    :param data_atlas:
    :param data_query:
    :param labels_atlas:
    :param labels_query:
    :param ks:
    :return:
    """
    if isinstance(ks, str):
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

