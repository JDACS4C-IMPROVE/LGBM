#!/bin/bash

# # Within-study
# python lgbm_infer_improve.py \
#     --test_ml_data_dir ml_data/GDSCv1-GDSCv1/split_0 \
#     --model_dir out_model/GDSCv1/split_0 \
#     --infer_outdir out_infer/GDSCv1-GDSCv1/split_0

# # Cross-study
# python lgbm_infer_improve.py \
#     --test_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
#     --model_dir out_model/GDSCv1/split_0 \
#     --infer_outdir out_infer/GDSCv1-CCLE/split_0

SPLIT=0

# ----------------------------------------
# Within-study analysis
# ----------------------------------------

# # Legacy
# SOURCE=CCLE
# TARGET=CCLE
# python lgbm_infer_improve.py \
#     --test_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_dir out_model/${SOURCE}/split_${SPLIT} \
#     --infer_outdir out_infer/${SOURCE}-${TARGET}/split_${SPLIT}

# # improvelib
# SOURCE=CCLE
# TARGET=CCLE
# python lgbm_infer_improve.py \
#     --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}

# ----------------------------------------
# Cross-study analysis
# ----------------------------------------

# # Legacy
# SOURCE=GDSCv1
# TARGET=CCLE
# python lgbm_infer_improve.py \
#     --test_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_dir out_model/${SOURCE}/split_${SPLIT} \
#     --infer_outdir out_infer/${SOURCE}-${TARGET}/split_${SPLIT}

# improvelib
SOURCE=GDSCv1
TARGET=CCLE
python lgbm_infer_improve.py \
    --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}

