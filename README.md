# mvTCR

## Install

`pip install mvtcr`

Or clone the reposetory via:

`git clone git@github.com:SchubertLab/mvTCR.git`

Then create a conda environment:

`conda create --name mvTCR python=3.11 -y && conda activate mvTCR && pip install -r requirements.txt && conda install nb_conda_kernels -y`

Afterwards install a suitable PyTorch version e.g. 2.1.2 with the appropriate CUDA Version following the instructions here: https://pytorch.org/get-started/previous-versions/

## Tutorial
We provide tutorials in the following notebooks:
- tutorials/01_preprocessing.ipynb : preprocessing the data, adding requiered annotation and filtering
- tutorials/02_training_analysis.ipynb : training the model, when you have some celltype and clonotype annotation
- tutorials/03_training_prediction.ipynb : training the model, when you have specificity annotation

## Computational Ressources
To train mvTCR a machine with GPU support is required. The hyperparameter optimization shown in our manuscript was performed on either a one single GPU machine with 32GB of memory or a 4-GPU node with 512GB of memory. In the latter case, the HPO can be parallelized to train 4 models simultaneously. The required memory scales with the dataset size. As a reference, 8GBs of RAM were sufficient to train mvTCR on our preprocessed 10x dataset (>60k cells).

## Reproducibility
To reproduce the results of the paper, please refer to: https://github.com/SchubertLab/mvTCR_reproducibility and mvTCR v0.1.3 : https://pypi.org/project/mvtcr/0.1.3/

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