from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "pypi_description.md").read_text()

setup(
    name='mvtcr',
    version='0.2.0',
    description='mvTCR: A multimodal generative model to learn a unified representation across TCR sequences and scRNAseq data for joint analysis of single-cell immune profiling data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Felix Drost, Yang An, Lisa M Dratva, Rik GH Lindeboom, Muzlifah Haniffa, Sarah A Teichmann, Fabian Theis, Mohammad Lotfollahi, Benjamin Schubert',
    maintainer='Felix Drost, Yang An, Irene Bonafonte Pardàs, Jan-Philipp Leusch',
    url='https://github.com/SchubertLab/mvTCR',
    packages=find_packages(include=['mvtcr', 'mvtcr.*']),
    install_requires=[
        'anndata==0.10.3',
        'comet-ml==3.35.5',
        'leidenalg==0.10.1',
        'muon==0.1.5',
        'numba==0.58.1',
        'numpy==1.26.1',
        'optuna==3.4.0',
        'pandas==2.1.1',
        'scanpy==1.9.6',
        'scikit-learn==1.3.2',
        'scirpy==0.13.1',
        'scrublet==0.2.3',
        'tqdm==4.66.1',
        'umap-learn==0.5.4'],
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
