#!/bin/bash

SPLIT=0

# ----------------------------------------
# Within-study analysis
# ----------------------------------------

# # Legacy
# SOURCE=CCLE
# TARGET=CCLE
# python lgbm_train_improve.py \
#     --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_outdir out_model/${SOURCE}/split_${SPLIT}

# improvelib
SOURCE=CCLE
TARGET=CCLE
python lgbm_train_improve.py \
    --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}

# ----------------------------------------
# Cross-study analysis
# ----------------------------------------

# # Legacy
# SOURCE=GDSCv1
# TARGET=CCLE
# python lgbm_train_improve.py \
#     --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_outdir out_model/${SOURCE}/split_${SPLIT}

# improvelib
SOURCE=GDSCv1
TARGET=CCLE
python lgbm_train_improve.py \
    --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}

