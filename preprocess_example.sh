#!/bin/bash

# Below are several examples of how to run the data preprocessing script.
# Currently, only the CSA runs are supported (within-study or cross-study).
# Uncomment and run the one you are you interested in.

# ----------------------------------------
# CSA (cross-study analysis) exmple
# ----------------------------------------

# Within-study
python lgbm_preprocess_improve.py \
    --train_split_file GDSCv1_split_4_train.txt \
    --val_split_file GDSCv1_split_4_val.txt \
    --test_split_file GDSCv1_split_4_test.txt \
    --ml_data_outdir ml_data/GDSCv1-GDSCv1/split_4

# # Cross-study
# python lgbm_preprocess_improve.py \
#     --train_split_file GDSCv1_split_4_train.txt \
#     --val_split_file GDSCv1_split_4_val.txt \
#     --test_split_file CCLE_all.txt \
#     --ml_data_outdir ml_data/GDSCv1-CCLE/split_4


# ----------------------------------------
# LCA (learning curve analysis) exmple
# ----------------------------------------

# # Train with sample size 1000
# python frm_preprocess_tr_vl_te.py \
#     --train_data_name CCLE \
#     --val_data_name CCLE \
#     --test_data_name CCLE \
#     --train_split_file_name CCLE_split_4_train_size_1024.txt \
#     --val_split_file_name CCLE_split_4_val.txt \
#     --test_split_file_name CCLE_split_4_test.txt \
#     --outdir csa_data/ml_data/CCLE/split_4/size_1000

# # Train with sample size 8000
# python frm_preprocess_tr_vl_te.py \
#     --train_data_name CCLE \
#     --val_data_name CCLE \
#     --test_data_name CCLE \
#     --train_split_file_name CCLE_split_4_train_size_8000.txt \
#     --val_split_file_name CCLE_split_4_val.txt \
#     --test_split_file_name CCLE_split_4_test.txt \
#     --outdir csa_data/ml_data/CCLE/split_4/size_8000


# ----------------------------------------
# Error analysis exmple
# ----------------------------------------

# # Train with ...
# python frm_preprocess_tr_vl_te.py \
#     --train_data_name CCLE \
#     --val_data_name CCLE \
#     --test_data_name CCLE \
#     --train_split_file_name CCLE_split_4_train.txt \
#     --val_split_file_name CCLE_split_4_val.txt \
#     --test_split_file_name CCLE_split_4_test.txt \
#     --outdir csa_data/ml_data/CCLE/split_4


# ----------------------------------------
# Robustness
# ----------------------------------------

# # Train with noise level of 2 added to x data
# python frm_preprocess_tr_vl_te.py \
#     --train_data_name CCLE \
#     --val_data_name CCLE \
#     --test_data_name CCLE \
#     --train_split_file_name CCLE_split_4_train.txt \
#     --val_split_file_name CCLE_split_4_val.txt \
#     --test_split_file_name CCLE_split_4_test.txt \
#     --outdir csa_data/ml_data/CCLE/split_4/x_noise_2

# # Train with noise level of 7 added to x data
# python frm_preprocess_tr_vl_te.py \
#     --train_data_name CCLE \
#     --val_data_name CCLE \
#     --test_data_name CCLE \
#     --train_split_file_name CCLE_split_4_train.txt \
#     --val_split_file_name CCLE_split_4_val.txt \
#     --test_split_file_name CCLE_split_4_test.txt \
#     --outdir csa_data/ml_data/CCLE/split_4/x_noise_7
