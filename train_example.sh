#!/bin/bash

# Below are several examples of how to run the data train script.
# Currently, only CSA runs are supported (within-study or cross-study).
# Uncomment and run the one you are you interested in.

SPLIT=0

# ----------------------------------------
# Within-study analysis
# ----------------------------------------

# # Legacy
# SOURCE=CCLE
# TARGET=$SOURCE
# python lgbm_train_improve.py \
#     --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_outdir out_model/${SOURCE}/split_${SPLIT}

# improvelib
SOURCE=CCLE
TARGET=$SOURCE
ML_DATA_DIR=./res/ml_data/${SOURCE}-${TARGET}/split_${SPLIT}
MODEL_DIR=./res/models/${SOURCE}/split_${SPLIT}
python lgbm_train_improve.py \
    --input_dir $ML_DATA_DIR \
    --output_dir $MODEL_DIR
    # --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    # --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}

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
ML_DATA_DIR=./res/ml_data/${SOURCE}-${TARGET}/split_${SPLIT}
MODEL_DIR=./res/models/${SOURCE}/split_${SPLIT}
python lgbm_train_improve.py \
    --input_dir $ML_DATA_DIR \
    --output_dir $MODEL_DIR
    # --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    # --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}
