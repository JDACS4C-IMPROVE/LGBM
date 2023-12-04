#!/bin/bash --login

set -e

# # Manually run these commands before running this sciprt
# conda env create --file environment.yml --force

# conda create -n lgbm_py37 python=3.7 pip --yes

# Not required (my vim env)
conda install -c conda-forge ipdb=0.13.9 --yes
conda install -c conda-forge python-lsp-server=1.2.4 --yes

# IMPROVE
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop # CANDLE

# Other
conda install -c conda-forge lightgbm --yes # LigthGBM
pip install pyarrow # saves and loads parquet files
