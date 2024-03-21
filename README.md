# mvTCR
[![PyPI version](https://img.shields.io/pypi/v/mvtcr.svg)](https://pypi.org/project/mvtcr)
[![PyPIDownloads](https://pepy.tech/badge/mvtcr)](https://pepy.tech/project/mvtcr)

## Install
### Linux and Windows

First create a new conda environment and install mvTCR via pip install:

`conda create --name mvTCR python=3.10 -y && conda activate mvTCR && pip install mvtcr && conda install nb_conda_kernels -y`

Then install PyTorch 2.1.0 with the correct CUDA Version following the command here: https://pytorch.org/get-started/previous-versions/

### Installation Note
The installation procedure will take approx. 10 minutes and has been tested on Windows 10 and Linux 4.18.0. For MAC with M1 chip, there are currently several dependencies not available. A machine with GPU- and CUDA-support is heavily encouraged.

## Tutorial
We provide tutorials in the following notebooks:
- tutorials/01_preprocessing.ipynb : adding requiered annotation and filtering on the data.
- tutorials/02_training_analysis.ipynb : training the model, when you have some cell type and clonotype annotation.
- tutorials/03_training_prediction.ipynb : training the model, when you have specificity annotation.

The notebooks are run on a subsampled dataset and can be executed within 15 minutes each. The expected output is provided in each notebook. 

## Computational Ressources
To train mvTCR a machine with GPU support is required. The hyperparameter optimization shown in our manuscript was performed on either a one single GPU machine with 32GB of memory or a 4-GPU node with 512GB of memory. In the latter case, the HPO can be parallelized to train 4 models simultaneously. The required memory scales with the dataset size. As a reference, 8GBs of RAM were sufficient to train mvTCR on our preprocessed 10x dataset (>60k cells).

## Reproducibility
To reproduce the results of the paper, please refer to: https://github.com/SchubertLab/mvTCR_reproducibility
The experiments were run with mvTCR v0.1.3 together with PyTorch v1.8.0.

## Reference 

If mvTCR is useful in your research, please cite:  
```
@article {Drost2021.06.24.449733,
	author = {Drost, Felix and An, Yang and Dratva, Lisa M and Lindeboom, Rik GH and Haniffa, Muzlifah and Teichmann, Sarah A and Theis, Fabian and Lotfollahi, Mohammad and Schubert, Benjamin},
	title = {Integrating T-cell receptor and transcriptome for large-scale single-cell immune profiling analysis},
	elocation-id = {2021.06.24.449733},
	year = {2022},
	doi = {10.1101/2021.06.24.449733},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/10/25/2021.06.24.449733},
	eprint = {https://www.biorxiv.org/content/early/2022/10/25/2021.06.24.449733.full.pdf},
	journal = {bioRxiv}
}

```