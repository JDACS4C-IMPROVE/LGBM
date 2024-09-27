#!/bin/bash --login

set -e

# Creating conda env with python 3.7 on Mac M1
# https://stackoverflow.com/questions/70205633/cannot-install-python-3-7-on-osx-arm64

# CONDA_ENV_NAME=lgbm_py37
CONDA_ENV_NAME=lgbm_py37_improve

# Create conda env
# conda create -n $CONDA_ENV_NAME python=3.7 pip lightgbm=3.1.1 pyarrow=12.0.1 --yes
conda create -n $CONDA_ENV_NAME python=3.7 pip lightgbm=3.1.1 --yes

# Activate conda env
conda activate $CONDA_ENV_NAME

# Packages
conda install conda-forge::pandas=1.3.0
conda install conda-forge::scikit-learn=1.0.2
conda install conda-forge::pyyaml=6.0
conda install conda-forge::pyarrow=9.0.0

# Install CANDLE
# pip install git+https://github.com/ECP-CANDLE/candle_lib@develop

# # Other 
# conda install -c conda-forge ipdb=0.13.9 --yes
# conda install -c conda-forge python-lsp-server=1.2.4 --yes
