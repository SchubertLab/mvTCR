"""
Here, we evalute the performance of TESSA from:
Zhang Z, Xiong D, Wang X, Liu H, Wang T.
Mapping the functional landscape of T cell receptor repertoires by single-T cell transcriptomics. Nat Methods. 2021.
https://www.nature.com/articles/s41592-020-01020-3
Clone the Github repository from https://github.com/jcao89757/TESSA to the folder 'baseline'.
To reproduce create a new environment based on the file 'baseline/requirement_tessa.txt' todo
Additionally, you must install the following R packages to the R distribution in your environment: todo
"""
import os
import pandas as pd
import numpy as np
import json
import shutil
import argparse

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def create_folders():
    """
    Create the base folders to store tmps and results
    :return: None
    """
    path_file = os.path.dirname(os.path.abspath(__file__))
    paths = ['/tmp', '/results']
    for p in paths:
        p = path_file + p
        if not os.path.exists(p):
            os.mkdir(p)


def run_tessa(dir_in, dir_out):
    """
    Runs the TESSA Encoder of a t cell repertoire
    :param dir_in: Path to the input directory
    :param dir_out: Path to the output directory
    :return: several output files in dir_out
    """
    path_file = os.path.dirname(os.path.abspath(__file__))
    settings_full = {
        'tcr': f'{dir_in}/tcrs_atlas.csv',
        'model': f'{path_file}/TESSA/BriseisEncoder/TrainedEncoder.h5',
        'embeding_vectors': f'{path_file}/TESSA/BriseisEncoder/Atchley_factors.csv',
        'output_TCR': f'{dir_out}/tessa_tcr_embedding.csv',
        'output_log': f'{dir_out}/tessa_log.log',
        'exp': f'{dir_in}/scRNA_atlas.csv',
        'output_tessa': f'{dir_out}/res/',
        'within_sample_networks': 'FALSE',

    }

    command_full = f'python {path_file}/TESSA/Tessa_main.py'
    for key, value in settings_full.items():
        command_full += f' -{key} {value}'

    os.system(command_full)


def run_briseis(dir_in, dir_out):
    """
    Runs the Briseis Encoder of a t cell repertoire
    :param dir_in: Path to the input directory
    :param dir_out: Path to the output directory
    :return: several output files in dir_out
    """
    path_file = os.path.dirname(os.path.abspath(__file__))
    settings_ae = {
        'tcr': f'{dir_in}/tcrs_query.csv',
        'model': f'{path_file}/TESSA/BriseisEncoder/TrainedEncoder.h5',
        'embeding_vectors': f'{path_file}/TESSA/BriseisEncoder/Atchley_factors.csv',
        'output_TCR': f'{dir_out}/tessa_tcr_embedding.csv',
        'output_log': f'{dir_out}/tessa_log.log',
    }
    command_ae = f'python {path_file}/TESSA/BriseisEncoder/BriseisEncoder.py'

    for key, value in settings_ae.items():
        command_ae += f' -{key} {value}'
    os.system(command_ae)


def get_labels(donor):
    """
    Extract the labels from the TCR file
    :param donor: id of the 10x donor
    :return: list [num_cells], list [num_cells] representing the binding labels for atlas and query set
    """
    path_file = os.path.dirname(os.path.abspath(__file__))
    path_labels = path_file + f'/../data/tessa/10x/{donor}/'
    df_atlas = pd.read_csv(path_labels+'tcrs_atlas.csv')
    df_query = pd.read_csv(path_labels+'tcrs_query.csv')
    return df_atlas['binding_name'].tolist(), df_query['binding_name'].tolist()


def get_tessa_weights(base_dir):
    """
    Load the b-values from the result RData
    :param base_dir: path to the base folder of the experiment
    :return: numpy array [3] giving the b-weights
    """
    rob.r['load'](f'{base_dir}/res/tessa_final.RData')
    b = rob.r['tessa_results'][0]
    b = np.array(b)
    return b


def get_tessa_unweighted_distances(base_dir):
    """
    Extract the Briseis encoding from file
    :param base_dir: path to the base folder of the experiment
    :return: numpy array [num_cells, 30] giving the embedding space by Briseis
    """
    unweighted_dist = pd.read_csv(f'{base_dir}/tessa_tcr_embedding.csv', index_col=0)
    unweighted_dist = unweighted_dist.values
    return unweighted_dist


def get_weighted_distances(unweighted_dist, b):
    """
    Calculates the weigthed distance
    :param unweighted_dist: numpy array [num_cells, 30] giving the TCR embedding
    :param b: numpy array [30] giving the position weights
    :return: numpy array [num_cells, 30] giving the weighted embedding
    """
    weighted_dist = unweighted_dist * b
    return weighted_dist


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


def save_dict(path, dictionary):
    """
    Saves a dict to a json file
    :param path: desired output path
    :param dictionary: dict to be saved
    :return: saves to file
    """
    with open(path, 'w') as fp:
        json.dump(dictionary, fp)


def run_evaluation(donor):
    """
    Runs the evaluation of TESSA for the specified donor
    :param donor: int, donor id
    :return: saves results and summary
    """
    path_file = os.path.dirname(os.path.abspath(__file__))
    dir_in = path_file + f'/../data/tessa/10x/{donor}/'
    dir_out_atlas = path_file + f'/tmp/{donor}/atlas'
    dir_out_query = path_file + f'/tmp/{donor}/query'

    if not os.path.exists(path_file + f'/tmp/{donor}/'):
        os.mkdir(path_file + f'/tmp/{donor}/')
    if not os.path.exists(dir_out_atlas):
        os.mkdir(dir_out_atlas)
    if not os.path.exists(dir_out_query):
        os.mkdir(dir_out_query)
    if os.path.exists(path_file + f'/tmp/{donor}/atlas/res'):
        shutil.rmtree(path_file + f'/tmp/{donor}/atlas/res')
    if os.path.exists(path_file + f'/tmp/{donor}/query/res'):
        shutil.rmtree(path_file + f'/tmp/{donor}/query/res')

    print('Run TESSA')
    run_tessa(dir_in, dir_out_atlas)
    print('Run Briseis')
    run_briseis(dir_in, dir_out_query)

    labels_atlas, labels_query = get_labels(donor)

    print('Calculate Briseis Performance')
    embedding_atlas = get_tessa_unweighted_distances(dir_out_atlas)
    embedding_query = get_tessa_unweighted_distances(dir_out_query)
    unweighted_results = get_knn_classification(embedding_atlas, embedding_query, labels_atlas, labels_query)
    save_dict(path_file+f'/results/{donor}_unweighted.json', unweighted_results)

    print('Calculate Tessa Performance')
    b = get_tessa_weights(dir_out_atlas)
    embedding_query = get_weighted_distances(embedding_query, b)
    embedding_atlas = get_weighted_distances(embedding_atlas, b)
    weighted_results = get_knn_classification(embedding_atlas, embedding_query, labels_atlas, labels_query)
    save_dict(path_file + f'/results/{donor}_weighted.json', weighted_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_env', type=str, default='C:/Users/felix.drost/Anaconda3/envs/tessa/Lib/R')
    parser.add_argument('--donor', type=str, default='test')
    args = parser.parse_args()

    path_env_r = args.path_env
    donor_tag = args.donor

    os.environ['R_HOME'] = path_env_r
    import rpy2.robjects as rob

    create_folders()
    run_evaluation(donor_tag)
