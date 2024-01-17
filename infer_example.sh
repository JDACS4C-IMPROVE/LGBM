#!/bin/bash

# # Within-study
# python lgbm_infer_improve.py \
#     --test_ml_data_dir ml_data/GDSCv1-GDSCv1/split_0 \
#     --model_dir out_model/GDSCv1/split_0 \
#     --infer_outdir out_infer/GDSCv1-GDSCv1/split_0

# Cross-study
python lgbm_infer_improve.py \
    --test_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
    --model_dir out_model/GDSCv1/split_0 \
    --infer_outdir out_infer/GDSCv1-CCLE/split_0
