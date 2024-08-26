#!/bin/bash --login
# Navigate to the dir with the cloned model repo
# Run it like this: source ./setup_improve.sh

# set -e

# Get current dir and model dir
model_path=$PWD
echo "Model path: $model_path"
model_name=$(echo "$model_path" | awk -F '/' '{print $NF}')
echo "Model name: $model_name"

# Download data (if needed)
data_dir="csa_data"
if [ ! -d $PWD/$data_dir/ ]; then
    echo "Download CSA data"
    source download_csa.sh
else
    echo "CSA data folder already exists"
fi

# Env var IMPROVE_DATA_DIR
export IMPROVE_DATA_DIR="./$data_dir/"

# Clone IMPROVE lib (if needed)
pushd ../
improve_lib_path=$PWD/IMPROVE
improve_branch="v0.0.3-beta"
if [ -d $improve_lib_path ]; then
  echo "IMPROVE repo exists in ${improve_lib_path}"
else
    # git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
    git clone -b $improve_branch https://github.com/JDACS4C-IMPROVE/IMPROVE
fi
pushd $model_name

# Env var PYTHOPATH
export PYTHONPATH=$PYTHONPATH:$improve_lib_path

echo
echo "IMPROVE_DATA_DIR: $IMPROVE_DATA_DIR"
echo "PYTHONPATH: $PYTHONPATH"
