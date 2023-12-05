""" Train LightGBM for drug response prediction.

Required outputs:
    * Trained model
    * Predictions on val data: val_scores.json
    * Prediction performance val scores: val_y_data_predicted.csv
    Everything in saved in params["model_outdir"]
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

import lightgbm as lgb

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics

# Model-specifc imports
from model_utils.utils import extract_subset_fea

# [Req] Imports from preprocess script
from lgbm_preprocess import preprocess_params

filepath = Path(__file__).resolve().parent # [Req]

# [Req] App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific args in the train script.
app_train_params = []

# [Req] Model-specific params (Model: LightGBM)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.
model_train_params = [
    {"name": "cuda_name",  # TODO. How should we control this?
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."
     },
    {"name": "learning_rate",
     "type": float,
     "default": 0.1,
     "help": "Learning rate for the optimizer."
    },
]

# [Req]
train_params = app_train_params + model_train_params
# req_train_args = ["model_outdir", "train_ml_data_dir", "val_ml_data_dir"]

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  


# [Req]
def run(params: Dict):
    """ Run model training.

    Args:
        params (dict): A dictionary of CANDLE/IMPROVE keywords and parsed values.

    Returns:
        dict: dict of prediction performance scores computed on
            validation data according to the metrics list.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir for the model
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["model_outdir"])
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")

    # ------------------------------------------------------
    # [Optional] Prepare dataloaders
    # ------------------------------------------------------
    tr_data = pd.read_parquet(Path(params["train_ml_data_dir"])/train_data_fname)
    vl_data = pd.read_parquet(Path(params["val_ml_data_dir"])/val_data_fname)

    fea_list = ["ge", "mordred"]
    fea_sep = "."
    xtr = extract_subset_fea(tr_data, fea_list=fea_list, fea_sep=fea_sep)
    ytr = tr_data[[params["y_col_name"]]]
    xvl = extract_subset_fea(vl_data, fea_list=fea_list, fea_sep=fea_sep)
    yvl = vl_data[[params["y_col_name"]]]
    print("xtr:", xtr.shape)
    print("xvl:", xvl.shape)
    print("ytr:", ytr.shape)
    print("yvl:", yvl.shape)

    # ------------------------------------------------------
    # [GraphDRP] Prepare, train, and save model
    # ------------------------------------------------------
    # Train model
    # import ipdb; ipdb.set_trace()
    ml_init_args = {'n_estimators': 1000, 'max_depth': -1,
                    'learning_rate': params["learning_rate"],
                    'num_leaves': 31, 'n_jobs': 8, 'random_state': None}
    ml_fit_args = {'verbose': False, 'early_stopping_rounds': 50}
    ml_fit_args['eval_set'] = (xvl, yvl)
    model = lgb.LGBMRegressor(objective='regression', **ml_init_args)
    model.fit(xtr, ytr, **ml_fit_args)

    # Save model
    # import ipdb; ipdb.set_trace()
    model.booster_.save_model(str(modelpath))

    # ------------------------------------------------------
    # [GraphDRP] Load best model and compute preditions
    # ------------------------------------------------------
    # import ipdb; ipdb.set_trace()
    # val_true, val_pred = evaluate_model(params, device, modelpath, val_loader)

    # Load the (best) saved model (as determined based on val data)
    # model = load_GraphDRP(params, modelpath, device)
    # model.eval()

    del model
    model = lgb.Booster(model_file=str(modelpath))

    # Compute predictions
    # val_true, val_pred = predicting(model, device, data_loader=val_loader) # (groud truth), (predictions)

    # Predict
    val_pred = model.predict(xvl)
    val_true = yvl.values.squeeze()
    # y_pred = model.predict(xvl)
    # y_true = yvl.values.squeeze()
    # pred = pd.DataFrame({"True": y_true, "Pred": y_pred})
    # vl_df = vl_data[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
    # pred = pd.concat([vl_df, pred], axis=1)
    # assert sum(pred["True"] == pred[trg_name]) == pred.shape[0], "Columns 'AUC' and 'True' are the ground truth."
   
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )

    return val_scores


# [Req]
# def main():
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params
    params = frm.initialize_parameters(
        filepath,
        # default_model="graphdrp_default_model.txt",
        # default_model="graphdrp_csa_params.txt",
        # default_model="params_ws.txt",
        # default_model="params_cs.txt",
        default_model="lgbm_params.txt",
        # default_model="lgb_params_ws.txt",
        # default_model="lgb_params_cs.txt",
        additional_definitions=additional_definitions,
        # required=req_train_args,
    )
    val_scores = run(params)
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    # main()
    main(sys.argv[1:])
