#!/bin/bash

# Below are several examples of how to run the data inference script.
# Currently, only CSA runs are supported (within-study or cross-study).
# Uncomment and run the one you are you interested in.

SPLIT=0

# ----------------------------------------
# Within-study analysis
# ----------------------------------------

# # Legacy
# SOURCE=CCLE
# TARGET=$SOURCE
# python lgbm_infer_improve.py \
#     --test_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_dir out_model/${SOURCE}/split_${SPLIT} \
#     --infer_outdir out_infer/${SOURCE}-${TARGET}/split_${SPLIT}

# improvelib
SOURCE=CCLE
TARGET=$SOURCE
ML_DATA_DIR=./res/ml_data/${SOURCE}-${TARGET}/split_${SPLIT}
MODEL_DIR=./res/models/${SOURCE}/split_${SPLIT}
INFER_DIR=./res/infer/${SOURCE}-${TARGET}/split_${SPLIT}
python lgbm_infer_improve.py \
    --input_data_dir $ML_DATA_DIR\
    --input_model_dir $MODEL_DIR\
    --output_dir $INFER_DIR
    # --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    # --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}

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
ML_DATA_DIR=./res/ml_data/${SOURCE}-${TARGET}/split_${SPLIT}
MODEL_DIR=./res/models/${SOURCE}/split_${SPLIT}
INFER_DIR=./res/infer/${SOURCE}-${TARGET}/split_${SPLIT}
python lgbm_infer_improve.py \
    --input_data_dir $ML_DATA_DIR\
    --input_model_dir $MODEL_DIR\
    --output_dir $INFER_DIR
    # --input_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT} \
    # --output_dir ./res/${SOURCE}-${TARGET}/split_${SPLIT}
