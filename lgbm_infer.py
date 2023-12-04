""" Functionality for inferencing with trained GraphDRP Model. """

import os
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from typing import Dict

from improve import framework as frm

# from improve.torch_utils import TestbedDataset
from improve.metrics import compute_metrics
# from models.gat import GATNet
# from models.gat_gcn import GAT_GCN
# from models.gcn import GCNNet
# from models.ginconv import GINConvNet

# # Model-specific imports
# from model_utils.torch_utils import (
#     TestbedDataset,
#     build_GraphDRP_dataloader,
#     # set_GraphDRP,
#     # train_epoch,
#     predicting,
#     load_GraphDRP,
#     determine_device
# )

# from graphdrp_train_improve import (
#     gdrp_data_conf,
#     gdrp_model_conf,
#     determine_device,
#     build_PT_data_loader,
#     evaluate_model,
#     store_predictions_df,
#     compute_performace_scores,
# )

# from graphdrp_train_improve import (
#     metrics_list,
#     model_preproc_params,
#     model_train_params,
#     # determine_device,
#     # build_PT_data_loader,
#     # evaluate_model,
#     # store_predictions_df,
#     # compute_performace_scores,
# )

from lgb_train import (
    metrics_list,
    model_preproc_params,
    model_train_params,
)

filepath = Path(__file__).resolve().parent

# [Req] App-specific params (App: monotherapy drug response prediction)
app_infer_params = []

# [GraphDRP] Model-specific params (Model: GraphDRP)
model_infer_params = [
    # {"name": "test_data_df",
    #  "default": frm.SUPPRESS,
    #  "type": str,
    #  "help": "Data frame with original test response data."
    # },

]

# req_infer_args = ["model_arch",
#                   # "model_outdir",
#                   "test_ml_data_dir",
#                   "model_dir",
#                   # "test_data_processed"
#                   "infer_outdir"
#                   ]


# def check_data_available(params: Dict) -> frm.DataPathDict:
#     """
#     Sweep the expected input paths and check that files needed in testing are available.

#     :param Dict params: Dictionary of parameters read

#     :return: Path to directories requested stored in dictionary with str key str and Path value.
#     :rtype: DataPathDict
#     """
#     # Expected
#     # test_ml_data_dir / processed / test_data_processed
#     # test_data_processed --> it has pt extension and is located inside a 'processed' folder
#     # Make sure that the test data exists
#     itestpath = Path(params["test_ml_data_dir"]) / "processed"
#     if itestpath.exists() == False:
#         raise Exception(f"ERROR ! Processed testing data folder {itestpath} not found.\n")
#     itest = itestpath / params["test_data_processed"]
#     if itest.exists() == False:
#         raise Exception(f"ERROR ! Processed testing data {itest} not found.\n")

#     # Check if testing data frame exists
#     itestdfpath = None
#     if "test_data_df" in params:
#         itestdfpath = Path(params["test_ml_data_dir"]) / params["test_data_df"]
#         if itestdfpath.exists() == False:
#             itestdfpath = None
#             message = (
#                        f"Data frame with original testing response data: {params['test_data_df']} not found." \
#                        + " Will continue but will only store partial (available) data frame.\n"
#             )
#             warnings.warn(message, RuntimeWarning)
#     else:
#         message = (
#                    f"Data frame with original testing response data not specified (no 'test_data_df' keyword)." \
#                        + " Will continue but will only store partial (available) data frame.\n"
#             )
#         warnings.warn(message, RuntimeWarning)

#     # Create output directory. Do not complain if it exists.
#     opath = Path(params["model_outdir"])
#     os.makedirs(opath, exist_ok=True)
#     modelpath = opath / params["model_params"]
#     if modelpath.exists() == False:
#         raise Exception(f"ERROR ! Trained model {modelpath} not found.\n")
#     fname = f"test_{params['model_eval_suffix']}.csv"
#     if itestdfpath is None:
#         fname = f"test_{params['model_eval_suffix']}_partial.csv"
#     predpath = opath / fname
#     fname = f"test_{params['json_scores_suffix']}.json"
#     scorespath = opath / fname

#     # Return in DataPathDict structure
#     inputdtd = {"test": itestpath, "model": modelpath, "df": itestdfpath}
#     outputdtd = {"pred": predpath, "scores": scorespath}
#     return inputdtd, outputdtd


# Get features (x), target (y), and meta
def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]


def run(params):
    """Execute specified model inference.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.

    :return: List of floats evaluating model predictions according to
             specified metrics.
    :rtype: float list
    """
    # import ipdb; ipdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir for the model. 
    # ------------------------------------------------------
    # import pdb; pdb.set_trace()
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # Check data availability and create output directory
    # ------------------------------------------------------
    # [Req]
    # indtd, outdtd = check_data_available(params)
    # indtd is dictionary with input_description: path components
    # outdtd is dictionary with output_description: path components

    # ------------------------------------------------------
    # [Req] Create data names for test
    # ------------------------------------------------------
    # test_data_fname = frm.build_ml_data_name(params, stage="test",
    #                                          file_format=params["data_format"])
    test_data_fname = frm.build_ml_data_name(params, stage="test")
    # GraphDRP -- remove data_format
    # test_data_fname = test_data_fname.split(params["data_format"])[0]

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

    # print("\nTest data:")
    # print(f"test_ml_data_dir: {params['test_ml_data_dir']}")
    # print(f"test_batch: {params['test_batch']}")
    # test_loader = build_GraphDRP_dataloader(params["test_ml_data_dir"],
    #                                         test_data_fname,
    #                                         params["test_batch"],
    #                                         shuffle=False)

    # -----------------------------
    # [GraphDRP] Load best model and compute preditions
    # -----------------------------
    # import ipdb; ipdb.set_trace()
    # test_true, test_pred = evaluate_model(params["model_arch"], device, indtd["model"], test_loader)
    # test_true, test_pred = evaluate_model(params, device, modelpath, test_loader)

    modelpath = frm.build_model_path(params, model_dir=params["model_dir"]) # [Req]

    # Load LightGBM
    import lightgbm as lgb
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
   

    # -----------------------------
    # [Req] Save raw predictions in dataframe
    # -----------------------------
    # import ipdb; ipdb.set_trace()
    frm.store_predictions_df(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )

    # -----------------------------
    # [Req] Compute performance scores
    # -----------------------------
    # import ipdb; ipdb.set_trace()
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores


# def main():
def main(args):
    # import ipdb; ipdb.set_trace()
    additional_definitions = model_preproc_params + \
                             model_train_params + \
                             model_infer_params + \
                             app_infer_params
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
        # required=req_infer_args,
        required=None,
    )
    # print("test_ml_data_dir:", params["test_ml_data_dir"])
    # print("model_dir:", params["model_dir"])
    # print("infer_outdir:", params["infer_outdir"])
    # import ipdb; ipdb.set_trace()
    test_scores = run(params)
    print("\nFinished inference of GraphDRP model.")


if __name__ == "__main__":
    # main()
    main(sys.argv[1:])
