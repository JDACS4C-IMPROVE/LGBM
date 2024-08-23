""" Train LightGBM for drug response prediction.

Required outputs
----------------
All the outputs from this train script are saved in params["model_outdir"].

1. Trained model.
   The model is trained with train data and validated with val data. The model
   file name and file format are specified, respectively by
   params["model_file_name"] and params["model_file_format"].
   For LightGBM, the saved model:
        model.txt

2. Predictions on val data. 
   Raw model predictions calcualted using the trained model on val data. The
   predictions are saved in val_y_data_predicted.csv

3. Prediction performance scores on val data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in val_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import lightgbm as lgb

# [Req] IMPROVE imports
# from improve import framework as frm
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Model-specifc imports
from model_utils.utils import extract_subset_fea

# [Req] Imports from preprocess script
from lgbm_preprocess_improve import preprocess_params

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Model-specific params (Model: LightGBM)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.
model_train_params = [
    {"name": "n_estimators",
     "type": int,
     "default": 1000,
     "help": "Number of estimators."
    },
    {"name": "max_depth",
     "type": int,
     "default": -1,
     "help": "Max depth."
    },
    # {"name": "learning_rate", # TODO it's already defined in improvelib
    #  "type": float,
    #  "default": 0.1,
    #  "help": "Learning rate for the optimizer."
    # },
    {"name": "num_leaves",
     "type": int,
     "default": 31,
     "help": "Number of leaves."
    },
]

# train_params = app_train_params + model_train_params
train_params = model_train_params
# ---------------------

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  


# [Req]
def run(params: Dict):
    """ Run model training.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # breakpoint()
    # from pprint import pprint; pprint(params);

    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set
    # performance scores
    # frm.create_outdir(outdir=params["model_outdir"]) # TODO cfg.initialize_parameters creates params['output_dir'] where the model will be stored

    # Build model path
    # modelpath = frm.build_model_path(params, model_dir=params["model_outdir"]) # AP
    modelpath = frm.build_model_path(params, model_dir=params["output_dir"]) # TODO instead of model_outdir

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    # tr_data = pd.read_parquet(Path(params["train_ml_data_dir"])/train_data_fname)
    # vl_data = pd.read_parquet(Path(params["val_ml_data_dir"])/val_data_fname)
    tr_data = pd.read_parquet(Path(params["input_dir"])/train_data_fname) # TODO explore input_dir and output_dir
    vl_data = pd.read_parquet(Path(params["input_dir"])/val_data_fname) # TODO explore input_dir and output_dir

    fea_list = ["ge", "mordred"]
    fea_sep = "."

    # Train data
    xtr = extract_subset_fea(tr_data, fea_list=fea_list, fea_sep=fea_sep)
    ytr = tr_data[[params["y_col_name"]]]
    print("xtr:", xtr.shape)
    print("ytr:", ytr.shape)

    # Val data
    xvl = extract_subset_fea(vl_data, fea_list=fea_list, fea_sep=fea_sep)
    yvl = vl_data[[params["y_col_name"]]]
    print("xvl:", xvl.shape)
    print("yvl:", yvl.shape)

    # ------------------------------------------------------
    # Prepare, train, and save model
    # ------------------------------------------------------
    # Prepare model and train settings
    ml_init_args = {"n_estimators": params["n_estimators"],
                    "max_depth": params["max_depth"],
                    "learning_rate": params["learning_rate"],
                    "num_leaves": params["num_leaves"],
                    "n_jobs": 8,
                    "random_state": None}
    model = lgb.LGBMRegressor(objective='regression', **ml_init_args)

    # Train model
    ml_fit_args = {'verbose': False, 'early_stopping_rounds': 50}
    ml_fit_args['eval_set'] = (xvl, yvl)
    model.fit(xtr, ytr, **ml_fit_args)

    # Save model
    model.booster_.save_model(str(modelpath))
    del model

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    model = lgb.Booster(model_file=str(modelpath))

    # Compute predictions
    val_pred = model.predict(xvl)
    val_true = yvl.values.squeeze()
   
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        # outdir=params["model_outdir"]
        outdir=params["output_dir"] # TODO explore input_dir and output_dir
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        # outdir=params["model_outdir"],
        outdir=params["output_dir"], # TODO explore input_dir and output_dir
        metrics=metrics_list
    )

    return val_scores


def initialize_parameters(params=None):
    """ Initialize parameters for model training.

    Returns:
        dict: dict of IMPROVE/CANDLE parameters and parsed values.
    """
    # [Req] Additional definitions
    # additional_definitions = preprocess_params + train_params
    additional_definitions = train_params

    # [Req] Initialize parameters
    params = frm.initialize_parameters(
        filepath,
        default_model="lgbm_params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )

    return params


# [Req]
def main(args):
    # [Req]
    cfg = DRPTrainConfig()
    # params = frm.initialize_parameters(filepath, default_model="lgbm_params.txt", additional_definitions=additional_definitions, required=None)
    # additional_definitions = preprocess_params + train_params
    additional_definitions = train_params
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="lgbm_params.txt",
        default_model=None,
        additional_cli_section=None,
        additional_definitions=additional_definitions,
        required=None)
    val_scores = run(params)
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
