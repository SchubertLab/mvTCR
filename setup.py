from pathlib import Path

from setuptools import setup, find_packages

long_description = Path('README.md').read_text('utf-8')

try:
    from scmulti import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''

setup(name='tcr_embedding',
      version='0.1.0',
      description='Joint latent embedding of TCR sequences and scRNA',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author=__author__,
      author_email=__email__,
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      )
