#!/bin/bash --login

set -e

# conda create -n lgbm_py37 python=3.7 pip --yes

# # Not required
# conda install -c conda-forge ipdb=0.13.9 --yes
# conda install -c conda-forge python-lsp-server=1.2.4 --yes

# IMPROVE
#pip install git+https://github.com/ECP-CANDLE/candle_lib@develop # CANDLE

# Other
#conda install -c conda-forge lightgbm=3.1.1 --yes # LigthGBM
#pip install pyarrow=12.0.1 # saves and loads parquet files

conda create -n lgbm_py37 python=3.7 pip lightgbm=3.1.1 --yes
conda activate lgbm_py37
pip install pyarrow==12.0.1
# IMPROVE
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop # CANDLE
