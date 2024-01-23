#!/bin/bash

# Below are two examples of how to run the end-to-end scripts:
# 1. Within-study analysis
# 2. Cross-study analysis
# Uncomment and run the one you are you interested in.


# Download the benchmark CSA data
wget --cut-dirs=8 -P ./ -nH -np -m https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/


# ----------------------------------------
# 1. Within-study
# ---------------

# # Preprocess
# # All preprocess outputs are saved in params["ml_data_outdir"]
# python lgbm_preprocess_improve.py \
#     --train_split_file GDSCv1_split_0_train.txt \
#     --val_split_file GDSCv1_split_0_val.txt \
#     --test_split_file GDSCv1_split_0_test.txt \
#     --ml_data_outdir ml_data/GDSCv1-GDSCv1/split_0

# # Train
# # All train outputs are saved in params["model_outdir"]
# python lgbm_train_improve.py \
#     --train_ml_data_dir ml_data/GDSCv1-GDSCv1/split_0 \
#     --val_ml_data_dir ml_data/GDSCv1-GDSCv1/split_0 \
#     --model_outdir out_model/GDSCv1/split_0

# # Infer
# # All infer outputs are saved in params["infer_outdir"]
# python lgbm_infer_improve.py \
#     --test_ml_data_dir ml_data/GDSCv1-GDSCv1/split_0 \
#     --model_dir out_model/GDSCv1/split_0 \
#     --infer_outdir out_infer/GDSCv1-GDSCv1/split_0


# ----------------------------------------
# 2. Cross-study
# --------------

# Preprocess
# All preprocess outputs are saved in params["ml_data_outdir"]
python lgbm_preprocess_improve.py \
    --train_split_file GDSCv1_split_0_train.txt \
    --val_split_file GDSCv1_split_0_val.txt \
    --test_split_file CCLE_all.txt \
    --ml_data_outdir ml_data/GDSCv1-CCLE/split_0

# Train
# All train outputs are saved in params["model_outdir"]
python lgbm_train_improve.py \
    --train_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
    --val_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
    --model_outdir out_model/GDSCv1/split_0

# Infer
# All infer outputs are saved in params["infer_outdir"]
python lgbm_infer_improve.py \
    --test_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
    --model_dir out_model/GDSCv1/split_0 \
    --infer_outdir out_infer/GDSCv1-CCLE/split_0
