""" Inference with LightGBM for drug response prediction.

Required outputs
----------------
All the outputs from this infer script are saved in params["infer_outdir"].

1. Predictions on test data.
   Raw model predictions calcualted using the trained model on test data. The
   predictions are saved in test_y_data_predicted.csv

2. Prediction performance scores on test data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in test_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import lightgbm as lgb

# [Req] IMPROVE imports
# from improve import framework as frm
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Model-specifc imports
from model_utils.utils import extract_subset_fea

# [Req] Imports from preprocess and train scripts
from lgbm_preprocess_improve import preprocess_params
from lgbm_train_improve import metrics_list, train_params

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Model-specific params (Model: LightGBM)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# infer_params = app_infer_params + model_infer_params
infer_params = model_infer_params
# ---------------------


# [Req]
def run(params: Dict):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # breakpoint()
    # from pprint import pprint; pprint(params);

    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    # frm.create_outdir(outdir=params["infer_outdir"]) # TODO cfg.initialize_parameters creates params['output_dir'] where the model will be stored

    # ------------------------------------------------------
    # [Req] Create data name for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    # te_data = pd.read_parquet(Path(params["test_ml_data_dir"])/test_data_fname) # AP
    if "input_data_dir" in params:
        te_data = pd.read_parquet(Path(params["input_data_dir"]) / test_data_fname)
    else:
        te_data = pd.read_parquet(Path(params["input_dir"]) / test_data_fname)

    fea_list = ["ge", "mordred"]
    fea_sep = "."

    # Test data
    xte = extract_subset_fea(te_data, fea_list=fea_list, fea_sep=fea_sep)
    yte = te_data[[params["y_col_name"]]]
    print("xte:", xte.shape)
    print("yte:", yte.shape)

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build model path
    # modelpath = frm.build_model_path(params, model_dir=params["model_dir"]) # AP
    if "input_model_dir" in params:
        modelpath = frm.build_model_path(params, model_dir=params["input_model_dir"])
    else:
        modelpath = frm.build_model_path(params, model_dir=params["input_dir"])

    # Load LightGBM
    model = lgb.Booster(model_file=str(modelpath))

    # Predict
    test_pred = model.predict(xte)
    test_true = yte.values.squeeze()

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        # outdir=params["infer_outdir"] # AP
        outdir=params["output_dir"] # TODO instead of infer_outdir
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performace_scores(
            params,
            y_true=test_true, y_pred=test_pred, stage="test",
            # outdir=params["infer_outdir"], # AP
            outdir=params["output_dir"], # TODO instead of infer_outdir
            metrics=metrics_list
        )

    return True


# [Req]
def main(args):
    # [Req]
    # additional_definitions = preprocess_params + train_params + infer_params
    additional_definitions = infer_params
    # params = frm.initialize_parameters(filepath, default_model="lgbm_params.txt", additional_definitions=additional_definitions, required=None)
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="lgbm_params.txt",
        default_model=None,
        additional_cli_section=None,
        additional_definitions=additional_definitions,
        required=None)
    status = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
