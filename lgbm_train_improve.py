""" Train LightGBM for drug response prediction.

Required outputs
----------------
All the outputs from this train script are saved in params["output_dir"].

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
   the true values for performance metrics. The scores are saved as json in
   val_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import lightgbm as lgb

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# Model-specifc imports
from model_params_def import train_params # [Req]
from model_utils.utils import extract_subset_fea

filepath = Path(__file__).resolve().parent # [Req]


# [Req]
def run(params: Dict) -> Dict:
    """ Run model training.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data.
    """
    # breakpoint()
    # from pprint import pprint; pprint(params);

    # ------------------------------------------------------
    # [Req] Build model path
    # ------------------------------------------------------
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"]
    )

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")  # [Req]

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    tr_data = pd.read_parquet(Path(params["input_dir"]) / train_data_fname)
    vl_data = pd.read_parquet(Path(params["input_dir"]) / val_data_fname)

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
    # python 3.7 (lightgbm 3.1.1)
    ml_fit_args = {'verbose': False, 'early_stopping_rounds': 50}
    # python 3.9 (lightgbm 4.6.0)
    # ml_fit_args = {} 
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
        y_true=val_true, 
        y_pred=val_pred, 
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=val_true, 
        y_pred=val_pred, 
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    return val_scores


# [Req]
def main(args):
    # [Req]
    additional_definitions = train_params
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="lgbm_params.txt",
        additional_definitions=additional_definitions)
    val_scores = run(params)
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
