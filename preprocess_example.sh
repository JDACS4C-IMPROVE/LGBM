#!/bin/bash

# Below are several examples of how to run the data preprocessing script.
# Currently, only the CSA runs are supported (within-study or cross-study).
# Uncomment and run the one you are you interested in.

SPLIT=0

# ----------------------------------------
# Within-study analysis
# ----------------------------------------

# SOURCE=CCLE
# TARGET=CCLE
# python lgbm_preprocess_improve.py \
#     --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
#     --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
#     --test_split_file ${TARGET}_split_${SPLIT}_test.txt \
#     --input_dir ./csa_data/raw_data \
#     --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}
#     # --ml_data_outdir ml_data/${SOURCE}-${TARGET}/split_${SPLIT}

# ----------------------------------------
# Cross-study analysis
# ----------------------------------------

SOURCE=GDSCv1
TARGET=CCLE
python lgbm_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_all.txt \
    --input_dir ./csa_data/raw_data \
    --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}
    # --ml_data_outdir ml_data/${SOURCE}-${TARGET}/split_${SPLIT}
