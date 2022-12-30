# mvTCR

## Install
Clone the reposetory via:

`git clone git@github.com:SchubertLab/mvTCR.git`

### Linux
`conda create --name mvTCR python=3.8.8 -y && conda activate mvTCR && pip install -r requirements.txt && conda install nb_conda_kernels -y`

### Windows
Please uncomment torch from the requirements.txt, i.e. write a # before torch
Then execute the line to install all the requirements except PyTorch
`conda create --name tcr python=3.8.8 -y && conda activate tcr && pip install -r requirements.txt && conda install nb_conda_kernels -y`

Then install PyTorch 1.8.0 with the correct CUDA Version following the command here: https://pytorch.org/get-started/previous-versions/

## Tutorial
We provide tutorials in the following notebooks:
- tutorials/01_preprocessing.ipynb : adding requiered annotation and filtering on the data
- tutorials/02_training_analysis.ipynb : training the model, when you have some celltype and clonotype annotation
- tutorials/03_training_prediction.ipynb : training the model, when you have specificity annotation

## Reproducability
To reproduce the results of the paper, please refer to: https://github.com/SchubertLab/mvTCR_reproducibility

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
