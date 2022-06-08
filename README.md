# mvTCR

## Install
Clone the reposetory via:

`git clone git@github.com:SchubertLab/mvTCR.git`

### Linux
`conda create --name mvTCR python=3.8.8 -y && conda activate mvTCR && pip install -r requirements.txt && conda install nb_conda_kernels -y && conda install bottleneck -y`

### Windows
Please uncomment torch from the requirements.txt, i.e. write a # before torch
Then execute the line to install all the requirements except PyTorch
`conda create --name tcr python=3.8.8 -y && conda activate tcr && pip install -r requirements.txt && conda install nb_conda_kernels -y && conda install bottleneck -y`

Then install PyTorch 1.8.0 with the correct CUDA Version following the command here: https://pytorch.org/get-started/previous-versions/

## Tutorial
We provide tutorials in the following notebooks:
- tutorials/01_preprocessing.ipynb : adding requiered annotation and filtering on the data
- tutorials/02_training_analysis.ipynb : training the model, when you have some celltype and clonotype annotation
- tutorials/03_training_prediction.ipynb : training the model, when you have specificity annotation

## Reproducability
This repo did undergo major refactoring. Please use "commit 3fabfae6ecb7c6ce0d605ea0ab289a89533ac523" to recreate the results from the ICML paper.
Further, we provide notebooks to reproduce the results from our paper in `experiments/10x_evaluate_models.ipynb` and `experiments/covid_evaluation.ipynb`. Please also refer to the Get Datasets section to retrieve and preprocess the data.

## Reference 

If mvTCR is useful in your research, please cite:  
```
@article{an2021jointly,
  title={Jointly learning T-cell receptor and transcriptomic information to decipher the immune response},
  author={An, Yang and Drost, Felix and Theis, Fabian and Schubert, Benjamin and Lotfollahi, Mohammad},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
