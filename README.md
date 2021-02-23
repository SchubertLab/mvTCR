## All-in-one-line:
### Create environment and install packages

`conda create --name tcr python=3.8 -y && conda activate tcr && conda install seaborn scikit-learn statsmodels numba pytables nb_conda_kernels -y && conda install -c conda-forge python-igraph leidenalg -y && pip install scanpy scirpy && conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y`

### Fix some bugs (on Windows):

`pip uninstall h5py -y && conda install h5py -y && conda install -c defaults intel-openmp -f -y`

## Step-by-step: 
### Create conda environment and install requirements

`conda create --name tcr python=3.8`

`conda install seaborn scikit-learn statsmodels numba pytables nb_conda_kernels`

`conda install -c conda-forge python-igraph leidenalg`

`pip install scanpy scirpy`

`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

### Fix some bugs on Windows
The `h5py` package is not detected properly, so we need to reinstall it:

`pip uninstall h5py`

`conda install h5py`

Also the intel-openmp package is missing:

`conda install -c defaults intel-openmp -f`

## Get Datasets
### Wu 2020 3k Toy dataset
Wu 2020 3k dataset is part of scirpy. Just run the `preprocessing/download_wu2020_3k_toy_dataset.ipynb`

### BCC Dataset
#### Option 1: Download the data in h5ad format directy
https://tumde-my.sharepoint.com/:f:/g/personal/yang_an_tum_de/EuAeGXist_BCsI0vdJo0T1QBKT54JyG7-A5iRMeGJnhfLA?e=zz9wmC

#### Option 2: Download raw data and preprocess
First download the raw data from here: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123813 (all 4 BCC files) 
Unzip them into `data/Yost_2018/`

Then run the `preprocessing/bcc_save_as_h5ad.ipynb` Notebook for preprocessing.

## Example
For a walkthrough of this API please refer to the following notebook `example/tcr_first_example.ipynb`