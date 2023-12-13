#!/bin/bash

# # Within-study
# python lgbm_train_improve.py \
#     --train_ml_data_dir ml_data/GDSCv1-GDSCv1/split_0 \
#     --val_ml_data_dir ml_data/GDSCv1-GDSCv1/split_0 \
#     --model_outdir out_model/GDSCv1/split_0

# Cross-study
python lgbm_train_improve.py \
    --train_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
    --val_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
    --model_outdir out_model/GDSCv1/split_0
