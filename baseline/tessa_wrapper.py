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
import shutil
import argparse


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


def run_model_atlas_query(dataset, donor=None, split=0):
    """
    Runs the evaluation of TESSA for the specified donor for a query and an atlas dataset
    :param donor: int, donor id
    :param dataset: str, folder name of the dataset (e.g. '10x')
    :return: saves results and summary
    """
    if donor is not None:
        donor = f'{donor}/'
    else:
        donor = ''

    path_file = os.path.dirname(os.path.abspath(__file__))
    dir_in = path_file + f'/../data/tessa/{dataset}/{donor}{split}/'
    dir_out_atlas = path_file + f'/tmp/{dataset}/{donor}{split}/atlas'
    dir_out_query = path_file + f'/tmp/{dataset}/{donor}{split}/query'

    if not os.path.exists(path_file + f'/tmp/{dataset}/'):
        os.mkdir(path_file + f'/tmp/{dataset}/')
    if not os.path.exists(path_file + f'/tmp/{dataset}/{donor}'):
        os.mkdir(path_file + f'/tmp/{dataset}/{donor}')
    if not os.path.exists(path_file + f'/tmp/{dataset}/{donor}{split}/'):
        os.mkdir(path_file + f'/tmp/{dataset}/{donor}{split}/')
    if not os.path.exists(dir_out_atlas):
        os.mkdir(dir_out_atlas)
    if not os.path.exists(dir_out_query):
        os.mkdir(dir_out_query)
    if os.path.exists(path_file + f'/tmp/{dataset}/{donor}{split}/atlas/res'):
        shutil.rmtree(path_file + f'/tmp/{dataset}/{donor}{split}/atlas/res')
    if os.path.exists(path_file + f'/tmp/{dataset}/{donor}{split}/query/res'):
        shutil.rmtree(path_file + f'/tmp/{dataset}/{donor}{split}/query/res')

    print('Run TESSA')
    run_tessa(dir_in, dir_out_atlas)
    print('Run Briseis')
    run_briseis(dir_in, dir_out_query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_env', type=str, default='')
    parser.add_argument('--donor', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='10x')
    parser.add_argument('--split', type=int, default=0)
    args = parser.parse_args()

    path_env_r = args.path_env
    donor_tag = args.donor
    dataset_name = args.dataset
    split = args.split

    create_folders()
    run_model_atlas_query(dataset_name, donor_tag, split)
