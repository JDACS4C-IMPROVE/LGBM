#!/bin/bash --login

set -e

# Creating conda env with python 3.7 on Mac M1
# https://stackoverflow.com/questions/70205633/cannot-install-python-3-7-on-osx-arm64

conda create -n lgbm_py37 python=3.7 pip lightgbm=3.1.1 --yes
conda activate lgbm_py37
pip install pyarrow==12.0.1  # save and load parquet files

# # CANDLE
# pip install git+https://github.com/ECP-CANDLE/candle_lib@develop

# # Not required
# conda install -c conda-forge ipdb=0.13.9 --yes
# conda install -c conda-forge python-lsp-server=1.2.4 --yes

