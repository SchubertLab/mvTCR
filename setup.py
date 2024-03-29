from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "pypi_description.md").read_text()

setup(
    name='mvtcr',
    version='0.1.3',
    description='mvTCR: A multimodal generative model to learn a unified representation across TCR sequences and scRNAseq data for joint analysis of single-cell immune profiling data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Felix Drost, Yang An, Lisa M Dratva, Rik GH Lindeboom, Muzlifah Haniffa, Sarah A Teichmann, Fabian Theis, Mohammad Lotfollahi, Benjamin Schubert',
    maintainer='Felix Drost, Yang An, Irene Bonafonte Pardàs, Jan-Philipp Leusch',
    url='https://github.com/SchubertLab/mvTCR',
    packages=find_packages(include=['tcr_embedding', 'tcr_embedding.*']),
    install_requires=[
        'scanpy==1.7.0',
        'scirpy>=0.7',
        'numba==0.52.0',
        'pandas==1.2.3',
        'anndata==0.8.0',
        'leidenalg==0.8.4',
        'comet-ml',
        'numpy==1.20.3',
        'scikit-learn==0.24.1',
        'scrublet==0.2.3',
        'tqdm',
        'optuna==2.10.1',
        'umap-learn==0.5.1',
        'matplotlib==3.6.3',
        'sqlalchemy==1.4.26',
        'dunamai==1.18.1'],
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Operating System :: Microsoft',
        'Operating System :: POSIX :: Linux',
        #'Operating System :: MacOS',
        'Environment :: GPU :: NVIDIA CUDA',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
)
