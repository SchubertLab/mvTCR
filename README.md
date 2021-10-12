This repo currently undergoes major refactoring. Please use "commit 3fabfae6ecb7c6ce0d605ea0ab289a89533ac523" in the mean time.

## Install all dependencies using this one-liner:
cd to the cloned directory and execute the following

### Linux
`conda create --name tcr python=3.8.8 -y && conda activate tcr && pip install -r requirements.txt && conda install nb_conda_kernels -y && conda install bottleneck -y`

### Windows
Please uncomment torch from the requirements.txt, i.e. write a # before torch
Then execute the line to install all the requirements except PyTorch
`conda create --name tcr python=3.8.8 -y && conda activate tcr && pip install -r requirements.txt && conda install nb_conda_kernels -y && conda install bottleneck -y`

Then install PyTorch 1.8.0 with the correct CUDA Version following the command here: https://pytorch.org/get-started/previous-versions/
## Get Datasets
### 10x
Download the raw data, save to `data/10x_CD8TC/patient_*` with * indicating the patient number and preprocess using `preprocessing/10x_preprocessing.ipynb`

The files for all four donors can be downloaded form here: https://support.10xgenomics.com/single-cell-vdj/datasets under the section `Application Note - A New Way of Exploring Immunity`

In particular the following files are needed:
- ``vdj_v1_hs_aggregated_donor*_filtered_feature_bc_matrix.h5``
- ``vdj_v1_hs_aggregated_donor*_all_contig_annotations.csv``
- ``vdj_v1_hs_aggregated_donor*_binarized_matrix.csv``


### BCC Dataset
Download raw data and preprocess
First download the raw data from here: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123813 (all 4 BCC files) 
Unzip them into `data/Yost_2018/`

Then run the `preprocessing/bcc_save_as_h5ad.ipynb` Notebook for preprocessing.

### Covid Dataset
Dataset containing SARS-CoV-2 T cells described in https://www.medrxiv.org/content/10.1101/2020.12.07.20245274v1

Later available under https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171037

Move the data into `data/Covid/`

Then run the `preprocessing/covid_preprocessing.ipynb` Notebook for preprocessing.

## Example
We provide an example on how to train new models using either a config file or using Optuna for automatic hyperparameter optimization under `experiments/10x_optuna_tutorial.ipynb`

Further, we provide notebooks to reproduce the results from our paper in `experiments/10x_evaluate_models.ipynb` and `experiments/covid_evaluation.ipynb`. The pretrained model weights can be downloaded from here: XXX
Please also refer to the Get Datasets section to retrieve and preprocess the data.

## Reference 

If mvTCR is useful in your research, please consider citing:  
```
@article{an2021jointly,
  title={Jointly learning T-cell receptor and transcriptomic information to decipher the immune response},
  author={An, Yang and Drost, Felix and Theis, Fabian and Schubert, Benjamin and Lotfollahi, Mohammad},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
