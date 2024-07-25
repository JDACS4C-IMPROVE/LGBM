#!/bin/bash

# Below are two examples of how to run the end-to-end scripts:
# 1. Within-study analysis
# 2. Cross-study analysis

# # Download the benchmark CSA data
# wget --cut-dirs=8 -P ./ -nH -np -m https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/

# ======================================================================
# Set env variables:
# 1. IMPROVE_DATA_DIR
# 2. IMPROVE lib
# TODO finish this
# -------------------
# current_dir=/lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP
current_dir=$PWD
echo "PWD: $current_dir"
# ======================================================================

SPLIT=0

# ----------------------------------------
# 1. Within-study
# ---------------

SOURCE=CCLE
TARGET=CCLE

# Preprocess (improvelib)
python lgbm_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_split_${SPLIT}_test.txt \
    --input_dir ./csa_data/raw_data \
    --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}
    # --ml_data_outdir ml_data/${SOURCE}-${TARGET}/split_${SPLIT}

# Train (improvelib)
python lgbm_train_improve.py \
    --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}

# Infer (improvelib)
python lgbm_infer_improve.py \
    --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}


# ----------------------------------------
# 2. Cross-study
# --------------

SOURCE=GDSCv1
TARGET=CCLE

# Preprocess (improvelib)
python lgbm_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_all.txt \
    --input_dir ./csa_data/raw_data \
    --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}
    # --ml_data_outdir ml_data/${SOURCE}-${TARGET}/split_${SPLIT}

# Train (improvelib)
python lgbm_train_improve.py \
    --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}

# Infer (improvelib)
python lgbm_infer_improve.py \
    --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}

