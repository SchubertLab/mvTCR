from setuptools import setup, find_packages

setup(
    name='mvtcr',
    version='0.1.0',
    packages=find_packages(include=['tcr_embedding', 'tcr_embedding.*']),
    install_requires=[
        'scanpy>=1.7.0', 
        'anndata>=0.7.5', 
        'numba>=0.52.0', 
        'pandas>=1.2.3',
        'numpy>=1.20.3', 
        'scikit-learn>=0.24.1', 
        'tqdm', 
        'umap-learn>=0.5.1', 
        'torch>=1.8.0'],
    extras_require={
        'paper': [
            'scanpy==1.7.0', 
            'scirpy==0.6.1', 
            'numba==0.52.0', 
            'pandas==1.2.3', 
            'anndata==0.7.5', 
            'leidenalg==0.8.4', 
            'comet-ml', 
            'numpy==1.20.3', 
            'scikit-learn==0.24.1', 
            'scrublet==0.2.3', 
            'tqdm', 
            'optuna', 
            'umap-learn==0.5.1', 
            'torch==1.8.0'
            ],
        'full': [
            'comet-ml', 
            'scrublet==0.2.3', 
            'optuna', 
            'scirpy==0.6.1', 
            'leidenalg>=0.8.4'
            ],
    }
)