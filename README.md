# mvTCR
[![PyPI version](https://img.shields.io/pypi/v/mvtcr.svg)](https://pypi.org/project/mvtcr)
[![PyPIDownloads](https://pepy.tech/badge/mvtcr)](https://pepy.tech/project/mvtcr)

## Install
### Linux and Windows

First create a new conda environment and install mvTCR via pip install:

`conda create --name mvTCR python=3.10 -y && conda activate mvTCR && pip install mvtcr && conda install nb_conda_kernels -y`

Then install PyTorch e.g. v2.1.2 with a fitting CUDA Version following the command here: https://pytorch.org/get-started/previous-versions/

### Installation Note
The installation procedure will take approx. 10 minutes and has been tested on Windows 10 and Linux 4.18.0. For MAC with M1 chip, there are currently several dependencies not available. A machine with GPU- and CUDA-support is heavily encouraged.

## Tutorial
We provide tutorials in the following notebooks:
- tutorials/01_preprocessing.ipynb : adding requiered annotation and filtering on the data.
- tutorials/02_training_analysis.ipynb : training the model, when you have some cell type and clonotype annotation.
- tutorials/02b_training_analysis_mudata.ipynb : a combination of tutorial 01 and 02 showing seemless MuData support.
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
@article {Drost2024,
	author = {Drost, Felix and An, Yang and Bonafonte-Pardàs, Irene and Dratva, Lisa M. and Lindeboom, Rik G. H. and Haniffa, Muzlifah and Teichmann, Sarah A. and Theis, Fabian and Lotfollahi, Mohammad and Schubert, Benjamin},
	title = {Multi-modal generative modeling for joint analysis of single-cell T cell receptor and gene expression data},
	year = {2024},
	doi = {10.1038/s41467-024-49806-9},
	publisher = {Nature Communications},
	URL = {https://doi.org/10.1038/s41467-024-49806-9},
	journal = {Nature Communications},
	volume = {15},
	number = {1},
	pages = {5577},
	abstract = {Recent advances in single-cell immune profiling have enabled the simultaneous measurement of transcriptome and T cell receptor (TCR) sequences, offering great potential for studying immune responses at the cellular level. However, integrating these diverse modalities across datasets is challenging due to their unique data characteristics and technical variations. Here, to address this, we develop the multimodal generative model mvTCR to fuse modality-specific information across transcriptome and TCR into a shared representation. Our analysis demonstrates the added value of multimodal over unimodal approaches to capture antigen specificity. Notably, we use mvTCR to distinguish T cell subpopulations binding to SARS-CoV-2 antigens from bystander cells. Furthermore, when combined with reference mapping approaches, mvTCR can map newly generated datasets to extensive T cell references, facilitating knowledge transfer. In summary, we envision mvTCR to enable a scalable analysis of multimodal immune profiling data and advance our understanding of immune responses.}
}
```