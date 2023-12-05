""" Inference with LightGBM drug response prediction.

Required outputs:
    * Predictions on test data: test_scores.json
    * Prediction performance test scores: test_y_data_predicted.csv
    Everything in saved in params["infer_outdir"]
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

# [Req] Imports from preprocess and train scripts
from lgbm_preprocess import preprocess_params
from lgbm_train import metrics_list, train_params

filepath = Path(__file__).resolve().parent # [Req]

# [Req] App-specific params (App: monotherapy drug response prediction)
app_infer_params = []

# [Req] Model-specific params (Model: LightGBM)
model_infer_params = []

# [Req]
infer_params = app_infer_params + model_infer_params
# req_infer_params = []


# [Req]
def run(params):
    """ Run model inference.

    Args:
        params (dict): A dictionary of CANDLE/IMPROVE keywords and parsed values.

    Returns:
        dict: dict of prediction performance scores computed on
            test data according to the metrics list.
    """
    # import ipdb; ipdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir for the inference results
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # [Req] Create data name for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    # ------------------------------------------------------
    # [GraphDRP] Prepare dataloaders
    # ------------------------------------------------------
    # te_data = pd.read_csv(Path(params["test_ml_data_dir"])/test_data_fname)
    te_data = pd.read_parquet(Path(params["test_ml_data_dir"])/test_data_fname)

    fea_list = ["ge", "mordred"]
    fea_sep = "."
    xte = extract_subset_fea(te_data, fea_list=fea_list, fea_sep=fea_sep)
    yte = te_data[[params["y_col_name"]]]
    print("xte:", xte.shape)
    print("yte:", yte.shape)

    # -----------------------------
    # [GraphDRP] Load best model and compute preditions
    # -----------------------------
    # import ipdb; ipdb.set_trace()
    # test_true, test_pred = evaluate_model(params["model_arch"], device, indtd["model"], test_loader)
    # test_true, test_pred = evaluate_model(params, device, modelpath, test_loader)

    modelpath = frm.build_model_path(params, model_dir=params["model_dir"]) # [Req]

    # Load LightGBM
    model = lgb.Booster(model_file=str(modelpath))

    # # Load the (best) saved model (as determined based on val data)
    # modelpath = frm.build_model_path(params, model_dir=params["model_dir"]) # [Req]
    # model = load_GraphDRP(params, modelpath, device)
    # model.eval()

    # Compute predictions
    # test_true, test_pred = predicting(model, device, data_loader=test_loader) # (groud truth), (predictions)

    # Predict
    test_pred = model.predict(xte)
    test_true = yte.values.squeeze()
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
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores


# [Req]
# def main():
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        # default_model="graphdrp_default_model.txt",
        # default_model="graphdrp_csa_params.txt",
        # default_model="params_ws.txt",
        # default_model="params_cs.txt",
        default_model="lgbm_params.txt",
        # default_model="lgbm_params_ws.txt",
        # default_model="lgbm_params_cs.txt",
        additional_definitions=additional_definitions,
        # required=req_infer_args,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    # main()
    main(sys.argv[1:])
