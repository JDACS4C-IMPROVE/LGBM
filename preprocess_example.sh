#!/bin/bash

# Below are several examples of how to run the data preprocessing script.
# Currently, only the CSA runs are supported (within-study or cross-study).
# Uncomment and run the one you are you interested in.

# ----------------------------------------
# CSA (cross-study analysis) exmple
# ----------------------------------------

# # Within-study
# python lgbm_preprocess_improve.py \
#     --train_split_file GDSCv1_split_0_train.txt \
#     --val_split_file GDSCv1_split_0_val.txt \
#     --test_split_file GDSCv1_split_0_test.txt \
#     --ml_data_outdir ml_data/GDSCv1-GDSCv1/split_0

# Cross-study
python lgbm_preprocess_improve.py \
    --train_split_file GDSCv1_split_0_train.txt \
    --val_split_file GDSCv1_split_0_val.txt \
    --test_split_file CCLE_all.txt \
    --ml_data_outdir ml_data/GDSCv1-CCLE/split_0
