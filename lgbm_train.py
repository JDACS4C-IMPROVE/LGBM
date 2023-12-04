""" Functionality for Training a GraphDRP Model. """

import os
import json
import sys
import warnings
from pathlib import Path
from pprint import pformat
from typing import Dict, Union

import numpy as np
import pandas as pd

# import torch
# from torch_geometric.data import DataLoader

# IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics
# from candle import build_pytorch_optimizer, get_pytorch_function, keras_default_config, CandleCkptPyTorch
# from candle import CandleCkptPyTorch

# Model-specific imports
# from model_utils.torch_utils import (
#     TestbedDataset,
#     build_GraphDRP_dataloader,
#     set_GraphDRP,
#     train_epoch,
#     predicting,
#     load_GraphDRP,
#     determine_device
# )
# from models.gat import GATNet
# from models.gat_gcn import GAT_GCN
# from models.gcn import GCNNet
# from models.ginconv import GINConvNet
# ---
# from model_utils.models.gat import GATNet
# from model_utils.models.gat_gcn import GAT_GCN
# from model_utils.models.gcn import GCNNet
# from model_utils.models.ginconv import GINConvNet

# from graphdrp_preprocess_improve import gdrp_data_conf  # ap
from lgb_preprocess import model_preproc_params, app_preproc_params, preprocess_params  # ap

filepath = Path(__file__).resolve().parent

# [Req] List of metrics names to be compute performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

# [Req] App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific args for the train script.
app_train_params = []

# [GraphDRP] Model-specific params (Model: GraphDRP)
model_train_params = [
    # {"name": "val_data_df",
    #  "default": frm.SUPPRESS,
    #  "type": str,
    #  "help": "Data frame with original validation response data."
    # },
    {"name": "model_arch",
     "default": "GINConvNet",
     "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
     "type": str,
     "help": "Model architecture to run."},
    {"name": "log_interval",
     "action": "store",
     "type": int,
     "help": "Interval for saving o/p"},
    {"name": "cuda_name",  # TODO. frm. How should we control this?
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."},
    {"name": "learning_rate",
     "type": float,
     "default": 0.0001,
     "help": "Learning rate for the optimizer."
    },
]

# req_train_args = ["model_arch", "model_outdir",
#                   "train_ml_data_dir", "val_ml_data_dir",
#                   # "train_data_processed", "val_data_processed"]
#                   # "train_ml_data_fname", "val_ml_data_fname"]
#                   ]


# TODO. consider moving to model_utils
def str2Class(str):
    """Get model class from model name (str)."""
    return globals()[str]()


# def check_data_available(params: Dict) -> frm.DataPathDict:
#     """
#     Sweep the expected input paths and check that files needed in training are available.

#     :param Dict params: Dictionary of CANDLE/IMPROVE parameters read.

#     :return: Path to directories requested stored in dictionary with str key str and Path value.
#     :rtype: DataPathDict
#     """
#     # Expected
#     # train_ml_data_dir / processed / train_data_processed
#     # train_data_processed --> it has pt extension and is located inside a 'processed' folder
#     # train_ml_data_dir / processed / train_data_processed
#     # train_data_processed --> it has pt extension and is located inside a 'processed' folder

#     # import pdb; pdb.set_trace()

#     # Make sure that the train data exists
#     itrainpath = Path(params["train_ml_data_dir"]) / "processed"
#     if itrainpath.exists() == False:
#         raise Exception(f"ERROR ! Processed training data folder {itrainpath} not found.\n")

#     # itrain = itrainpath / params["train_data_processed"]
#     itrain = itrainpath / (params["train_ml_data_fname"] + ".pt")
#     if itrain.exists() == False:
#         raise Exception(f"ERROR ! Processed training data {itrain} not found.\n")

#     # Make sure that the val data exists
#     ivalpath = Path(params["val_ml_data_dir"]) / "processed"
#     if ivalpath.exists() == False:
#         raise Exception(f"ERROR ! Processed validation data folder {ivalpath} not found.\n")

#     # ival = ivalpath / params["val_data_processed"]
#     ival = ivalpath / (params["val_ml_data_fname"] + ".pt")
#     if ival.exists() == False:
#         raise Exception(f"ERROR ! Processed validation data {ival} not found.\n")

#     # Check if validation data frame exists
#     ivaldfpath = None
#     if "val_data_df" in params:
#         ivaldfpath = Path(params["val_ml_data_dir"]) / params["val_data_df"]
#         if ivaldfpath.exists() == False:
#             ivaldfpath = None
#             message = (
#                        f"Data frame with original validation response data: {params['val_data_df']} not found." \
#                        + " Will continue but will only store partial (available) data frame.\n"
#             )
#             warnings.warn(message, RuntimeWarning)
#     else:
#         message = (
#                    f"Data frame with original validation response data not specified (no 'val_data_df' keyword)." \
#                        + " Will continue but will only store partial (available) data frame.\n"
#             )
#         warnings.warn(message, RuntimeWarning)

#     # Create output directory. Do not complain if it exists.
#     opath = Path(params["model_outdir"])
#     os.makedirs(opath, exist_ok=True)
#     modelpath = opath / params["model_params"]

#     # import pdb; pdb.set_trace()

#     fname = f"val_{params['model_eval_suffix']}.csv"
#     if ivaldfpath is None:
#         fname = f"val_{params['model_eval_suffix']}_partial.csv"
#     predpath = opath / fname

#     fname = f"val_{params['json_scores_suffix']}.json"
#     scorespath = opath / fname

#     # Return in DataPathDict structure
#     inputdtd = {"train": itrainpath, "val": ivalpath, "df": ivaldfpath}
#     outputdtd = {"model": modelpath, "pred": predpath, "scores": scorespath}
#     return inputdtd, outputdtd


# def check_train_data_available(params: Dict) -> frm.DataPathDict:
#     """
#     Sweep the expected input paths and check that files needed in training are available.

#     :param Dict params: Dictionary of CANDLE/IMPROVE parameters read.

#     :return: Path to directories requested stored in dictionary with str key str and Path value.
#     :rtype: DataPathDict
#     """
#     # Expected
#     # train_ml_data_dir / processed / train_data_processed
#     # train_data_processed --> it has pt extension and is located inside a 'processed' folder
#     # train_ml_data_dir / processed / train_data_processed
#     # train_data_processed --> it has pt extension and is located inside a 'processed' folder

#     # import pdb; pdb.set_trace()

#     # Make sure that the train data exists
#     itrainpath = Path(params["train_ml_data_dir"])
#     if itrainpath.exists() == False:
#         raise Exception(f"ERROR ! Processed training data folder {itrainpath} not found.\n")

#     # Make sure that the val data exists
#     ivalpath = Path(params["val_ml_data_dir"])
#     if ivalpath.exists() == False:
#         raise Exception(f"ERROR ! Processed validation data folder {ivalpath} not found.\n")

#     train_data_fname = frm.build_ml_data_name(params, stage="train",
#                                               data_format=params["data_format"])
#     itrain = itrainpath / train_data_fname
#     if itrain.exists() == False:
#         raise Exception(f"ERROR ! Processed training data {itrain} not found.\n")

#     val_data_fname = frm.build_ml_data_name(params, stage="val",
#                                             data_format=params["data_format"])
#     ival = ivalpath / val_data_fname
#     if ival.exists() == False:
#         raise Exception(f"ERROR ! Processed validation data {ival} not found.\n")

#     # [Req] Save y data dataframe for the current stage
#     train_data_df = f"train_{params['y_data_suffix']}.csv"
#     val_data_df = f"val_{params['y_data_suffix']}.csv"
#     train_data_df_path = itrainpath / f"train_{params['y_data_suffix']}.csv"
#     val_data_df_path = ivalpath /f"val_{params['y_data_suffix']}.csv"

#     # Check if validation data frame exists
#     ivaldfpath = None
#     if "val_data_df" in params:
#         ivaldfpath = Path(params["val_ml_data_dir"]) / params["val_data_df"]
#         if ivaldfpath.exists() == False:
#             ivaldfpath = None
#             message = (
#                        f"Data frame with original validation response data: {params['val_data_df']} not found." \
#                        + " Will continue but will only store partial (available) data frame.\n"
#             )
#             warnings.warn(message, RuntimeWarning)
#     else:
#         message = (
#                    f"Data frame with original validation response data not specified (no 'val_data_df' keyword)." \
#                        + " Will continue but will only store partial (available) data frame.\n"
#             )
#         warnings.warn(message, RuntimeWarning)

#     # Create output directory. Do not complain if it exists.
#     opath = Path(params["model_outdir"])
#     os.makedirs(opath, exist_ok=True)
#     modelpath = opath / params["model_params"]

#     # import pdb; pdb.set_trace()

#     fname = f"val_{params['model_eval_suffix']}.csv"
#     if ivaldfpath is None:
#         fname = f"val_{params['model_eval_suffix']}_partial.csv"
#     predpath = opath / fname

#     fname = f"val_{params['json_scores_suffix']}.json"
#     scorespath = opath / fname

#     # Return in DataPathDict structure
#     inputdtd = {"train": itrainpath, "val": ivalpath, "df": ivaldfpath}
#     outputdtd = {"model": modelpath, "pred": predpath, "scores": scorespath}

#     return inputdtd, outputdtd


def config_checkpointing(params: Dict, model, optimizer):
    """Configure CANDLE checkpointing. Reads last saved state if checkpoints exist.

    :params str ckpt_directory: String with path to directory for storing the CANDLE checkpointing for the model being trained.

    :return: Number of training iterations already run (this may be > 0 if reading from checkpointing).
    :rtype: int
    """
    # params["ckpt_directory"] = ckpt_directory
    initial_epoch = 0
    # TODO. This creates directory self.params["ckpt_directory"]
    # import pdb; pdb.set_trace()
    ckpt = CandleCkptPyTorch(params)
    ckpt.set_model({"model": model, "optimizer": optimizer})
    J = ckpt.restart(model)
    if J is not None:
        initial_epoch = J["epoch"]
        print("restarting from ckpt: initial_epoch: %i" % initial_epoch)
    return ckpt, initial_epoch


# Considers Ray (not used)
# def train_graphdrp(params: Dict):
#     """User-defined training function that runs on each distributed worker process.

#     This function typically contains logic for loading the model,
#     loading the dataset, training the model, saving checkpoints,
#     and logging metrics.
#     """
#     # Model, Loss, Optimizer
#     # __init__()
#     model_arch = params["model_arch"]
#     model = str2Class(model_arch).to(device)
#     device = determine_device(params["cuda_name"])
#     # setup_train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     # loss_fn = get_pytorch_function(params["loss"])
#     loss_fn = torch.nn.MSELoss()

#     # Data (Prepare PyTorch dataloaders)
#     # -----------------------------
#     # Prepare PyTorch dataloaders
#     print("\nTraining data:")
#     print(f"train_ml_data_dir: {params['train_ml_data_dir']}")
#     print(f"train_ml_data_fname: {params['train_ml_data_fname']}")
#     print(f"batch_size: {params['batch_size']}")
#     # import pdb; pdb.set_trace()
#     train_loader = build_GraphDRP_dataloader(
#         params["train_ml_data_dir"],
#         # params["train_data_processed"],
#         params["train_ml_data_fname"],
#         params["batch_size"],
#         shuffle=True)

#     # Note! Don't shuffle the val_loader or results will be corrupted
#     print("\nVal data:")
#     print(f"val_ml_data_dir: {params['val_ml_data_dir']}")
#     print(f"val_ml_data_fname: {params['val_ml_data_fname']}")
#     print(f"val_batch: {params['val_batch']}")
#     # import pdb; pdb.set_trace()
#     val_loader = build_GraphDRP_dataloader(
#         params["val_ml_data_dir"],
#         params["val_ml_data_fname"],
#         params["val_batch"],
#         shuffle=False)

#     # Train. Prep settings.
#     if params["ckpt_directory"] is None:
#         params["ckpt_directory"] = params["model_outdir"]
#     # import ipdb; ipdb.set_trace()
#     # initial_epoch = trobj.config_checkpointing(params["ckpt_directory"])
#     initial_epoch = config_checkpointing(params, model, optimizer)

#     num_epoch = params["epochs"]
#     log_interval = params["log_interval"]
#     patience = params["patience"]
#     model.train()

#     # Settings for early stop and best model settings
#     best_score = np.inf
#     best_epoch = -1
#     early_stop_counter = 0  # define early-stop counter
#     # early_stop_metric = "mse"  # metric for early stop

#     # Train. Iterate over epochs.
#     for epoch in range(initial_epoch, num_epoch):
#         train_loss = train(train_loader, epoch + 1)
#         ckpt.ckpt_epoch(epoch, train_loss) # checkpoints the best model by default

#         # Predict with val data
#         val_true, val_pred = predicting(model, device, val_loader)  # val_true (groud truth), val_pred (predictions)
#         val_scores = compute_metrics(val_true, val_pred, metrics_list)

#         # For early stop
#         if val_scores[early_stop_metric] < best_score:
#             torch.save(model.state_dict(), modelpath)
#             best_epoch = epoch + 1
#             best_score = val_scores[early_stop_metric]
#             print(f"{early_stop_metric} improved at epoch {best_epoch}; Best \
#               {early_stop_metric}: {best_score}; Model: {model_arch}")
#             early_stop_counter = 0  # zero the early-stop counter if the model improved after the epoch
#         else:
#             print(f"No improvement since epoch {best_epoch}; Best \
#               {early_stop_metric}: {best_score}; Model: {model_arch}")
#             early_stop_counter += 1  # increment the counter if the model was not improved after the epoch

#         if early_stop_counter == patience:
#             print(f"Terminate training (model did not improve on val data for {self.patience} epochs).")
#             continue
#     return model


class Trainer:
    """Class to define a PyTorch interface for training models."""
    def __init__(self, params, device, modelpath, metrics_list=None):
        """Initialize a Trainer object.

        :params Dict params: Dictionary of CANDLE/IMPROVE parameters read.
        :params str device: String with PyTorch format describing device available for training.
        :params Path modelpath: Path to store model. Currently this is complementary
                to checkpointing, i.e. models are saved directly and also with CANDLE
                checkpointing. This redundacy should be re-evaluated.
        :params List metrics_list: List of strings specifying the functions to evaluate the
                model, e.g. "mse", "pcc", etc. Default: None which is converted to:
                ["mse", "rmse", "pcc", "scc"].
        """
        # -----------------------------
        # Create and move model to device
        self.params = params
        self.model_arch = params["model_arch"]
        self.model = str2Class(self.model_arch).to(device)
        self.device = device
        self.modelpath = modelpath
        if metrics_list is not None:  # TODO: not sure if need this.
            self.metrics_list = metrics_list
        else:
            self.metrics_list = ["mse", "rmse", "pcc", "scc"]

    def setup_train(self,):
        """Configure the Trainer object.

        This function constructs the optimizer and loss and extract
        trainer parameters for number of epochs, interval of logging
        model evaluation and patience for early stopping.
        """
        # Construct DL optimizer and loss
        keras_defaults = keras_default_config()  # TODO: keras?
        # TODO: build_pytorch_optimizer() supports a limited number of optimizers.
        # What if the model needs a different optimizer?
        self.optimizer = build_pytorch_optimizer(
            model=self.model,
            optimizer=self.params["optimizer"],
            lr=self.params["learning_rate"],
            kerasDefaults=keras_defaults
        )
        self.loss_fn = get_pytorch_function(self.params["loss"])
        # Train parameters
        self.num_epoch = self.params["epochs"]
        self.log_interval = self.params["log_interval"]
        self.patience = self.params["patience"]

    def config_checkpointing(self, ckpt_directory):
        """Configure CANDLE checkpointing. Reads last saved state if checkpoints exist.

        :params str ckpt_directory: String with path to directory for storing the CANDLE checkpointing for the model being trained.

        :return: Number of training iterations already run (this may be > 0 if reading from checkpointing).
        :rtype: int
        """
        self.params["ckpt_directory"] = ckpt_directory
        initial_epoch = 0
        # TODO. This creates directory self.params["ckpt_directory"]
        # import pdb; pdb.set_trace()
        self.ckpt = CandleCkptPyTorch(self.params)
        self.ckpt.set_model({"model": self.model, "optimizer": self.optimizer})
        J = self.ckpt.restart(self.model)
        if J is not None:
            initial_epoch = J["epoch"]
            print("restarting from ckpt: initial_epoch: %i" % initial_epoch)

        return initial_epoch

    def train(self, train_loader, epoch):
        """ Execute a training epoch (i.e. one pass through training set).

        :params DataLoader train_loader: PyTorch data loader with training data.
        :params int epoch: Current training epoch (for display purposes only).

        :return: Average loss for executed training epoch.
        :rtype: float
        """
        print("Training on {} samples...".format(len(train_loader.dataset)))
        self.model.train()
        # Below is the train() from the original GraphDRP model
        avg_loss = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output, _ = self.model(data)
            loss = self.loss_fn(output, data.y.view(-1, 1).float().to(self.device))
            loss.backward()
            self.optimizer.step()
            avg_loss.append(loss.item())
            if batch_idx % self.log_interval == 0:
                print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data.x),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        return sum(avg_loss) / len(avg_loss)

    def execute_train(self, train_loader, val_loader, initial_epoch, early_stop_metric="mse"):
        """ Execute a training loop, i.e. perform multiple passes through training set,
        updating model parameters, storing them (directly and via CANDLE checkpointing),
        evaluating model and enacting early stopping (if applicable, i.e. if for a
        `patience` number of iterations the selected metric does not improve when evaluating
        over the validation set).

        :params DataLoader train_loader: PyTorch data loader with training data.
        :params DataLoader val_loader: PyTorch data loader with validation data. Used for early stopping.
        :params int initial_epoch: Number of training iterations already run (this may be > 0 if reading from checkpointing).
        """

        # Settings for early stop and best model settings
        best_score = np.inf
        best_epoch = -1
        early_stop_counter = 0  # define early-stop counter
        # early_stop_metric = "mse"  # metric for early stop

        # Iterate over epochs
        for epoch in range(initial_epoch, self.num_epoch):
            train_loss = self.train(train_loader, epoch + 1)
            self.ckpt.ckpt_epoch(epoch, train_loss) # checkpoints the best model by default

            # Predict with val data
            val_true, val_pred = predicting(self.model, self.device, val_loader)  # val_true (groud truth), val_pred (predictions)
            val_scores = compute_metrics(val_true, val_pred, self.metrics_list)

            # For early stop
            if val_scores[early_stop_metric] < best_score:
                torch.save(self.model.state_dict(), self.modelpath)
                best_epoch = epoch + 1
                best_score = val_scores[early_stop_metric]
                print(f"{early_stop_metric} improved at epoch {best_epoch}; Best \
                  {early_stop_metric}: {best_score}; Model: {self.model_arch}")
                early_stop_counter = 0  # zero the early-stop counter if the model improved after the epoch
            else:
                print(f"No improvement since epoch {best_epoch}; Best \
                  {early_stop_metric}: {best_score}; Model: {self.model_arch}")
                early_stop_counter += 1  # increment the counter if the model was not improved after the epoch

            if early_stop_counter == self.patience:
                print(f"Terminate training (model did not improve on val data for {self.patience} epochs).")
                continue


# def evaluate_model(params, device, modelpath, data_loader):
#     """Load the model and perform predictions using given model.

#     :params str model_arch: Name of model architecture to use.
#     :params str device: Device to use for evaluating PyTorch model.
#     :params Path modelpath: Path containing model parameters.
#     :params DataLoader data_loader: PyTorch data loader with data to
#             use for evaluation.

#     :return: Arrays with ground truth and model predictions.
#     :rtype: np.array
#     """
#     # Load model
#     model = str2Class(params["model_arch"]).to(device)
#     model.load_state_dict(torch.load(modelpath))
#     model.eval()
#     # Compute predictions
#     val_true, val_pred = predicting(model, device, data_loader)  # (groud truth), (predictions)
#     return val_true, val_pred


# Get features (x), target (y), and meta
def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]
    

def run(params):
    """ Execute specified model training.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.

    :return: List of floats evaluating model predictions according to
             specified metrics_list.
    :rtype: float list
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir for the model. 
    # ------------------------------------------------------
    # import pdb; pdb.set_trace()
    # modelpath = frm.create_model_outpath(params)
    frm.create_outdir(outdir=params["model_outdir"])
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

    # ------------------------------------------------------
    # Check data availability and create output directory
    # ------------------------------------------------------
    # import pdb; pdb.set_trace()
    # indtd, outdtd = check_data_available(params)
    # indtd is dictionary with input_description: path components
    # outdtd is dictionary with output_description: path components

    # indtd, outdtd = check_train_data_available(params)

    # ------------------------------------------------------
    # [Req] Create data names for train and val
    # ------------------------------------------------------
    # train_data_fname = frm.build_ml_data_name(params, stage="train",
    #                                           file_format=params["data_format"])
    # val_data_fname = frm.build_ml_data_name(params, stage="val",
    #                                         file_format=params["data_format"])
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")
    # GraphDRP -- remove data_format
    # train_data_fname = train_data_fname.split(params["data_format"])[0]
    # val_data_fname = val_data_fname.split(params["data_format"])[0]

    # ------------------------------------------------------
    # [GraphDRP] Prepare dataloaders
    # ------------------------------------------------------
    # import ipdb; ipdb.set_trace()
    # tr_data = pd.read_csv(Path(params["train_ml_data_dir"])/train_data_fname)
    # vl_data = pd.read_csv(Path(params["val_ml_data_dir"])/val_data_fname)
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

    # import ipdb; ipdb.set_trace()
    # ydf_fname = f"{stage}_{params['y_data_suffix']}.csv"
    # ydf_fpath = Path(params[f"{stage}_ml_data_dir"]) / ydf_fname
    # rsp_df = pd.read_csv(ydf_fpath)
    # rsp_df[[params["y_col_name"]]]
    # v2 = pred_df[true_col_name].values.astype(np.float32)

    # print("\nTraining data:")
    # print(f"train_ml_data_dir: {params['train_ml_data_dir']}")
    # print(f"batch_size: {params['batch_size']}")
    # train_loader = build_GraphDRP_dataloader(params["train_ml_data_dir"],
    #                                          train_data_fname,
    #                                          params["batch_size"],
    #                                          shuffle=True)

    # # Don't shuffle the val_loader, otherwise results will be corrupted
    # print("\nVal data:")
    # print(f"val_ml_data_dir: {params['val_ml_data_dir']}")
    # print(f"val_batch: {params['val_batch']}")
    # val_loader = build_GraphDRP_dataloader(params["val_ml_data_dir"],
    #                                        val_data_fname,
    #                                        params["val_batch"],
    #                                        shuffle=False)

    # ------------------------------------------------------
    # [GraphDRP] CUDA/CPU device
    # ------------------------------------------------------
    # Determine CUDA/CPU device and configure CUDA device if available
    # TODO. How this should be configured with our (Singularity) workflows?
    # device = determine_device(params["cuda_name"])

    # ------------------------------------------------------
    # [GraphDRP] Prepare model
    # ------------------------------------------------------
    # Model, Loss, Optimizer
    # model = set_GraphDRP(params, device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    # loss_fn = torch.nn.MSELoss()

    # Train model
    # import ipdb; ipdb.set_trace()
    import lightgbm as lgb
    ml_init_args = {'n_estimators': 1000, 'max_depth': -1, 'learning_rate': 0.1,
                    'num_leaves': 31, 'n_jobs': 8, 'random_state': None}
    ml_fit_args = {'verbose': False, 'early_stopping_rounds': 50}
    ml_fit_args['eval_set'] = (xvl, yvl)
    model = lgb.LGBMRegressor(objective='regression', **ml_init_args)
    model.fit(xtr, ytr, **ml_fit_args)

    # Save model
    # import ipdb; ipdb.set_trace()
    model.booster_.save_model(str(modelpath))

    # -----------------------------
    # [GraphDRP] Load best model and compute preditions
    # -----------------------------
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
   
    # -----------------------------
    # [Req] Save raw predictions in dataframe
    # -----------------------------
    # import ipdb; ipdb.set_trace()
    frm.store_predictions_df(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"]
    )

    # -----------------------------
    # [Req] Compute performance scores
    # -----------------------------
    # import ipdb; ipdb.set_trace()
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )

    return val_scores


# def main():
def main(args):
    # import ipdb; ipdb.set_trace()
    additional_definitions = model_preproc_params + \
                             model_train_params + \
                             app_train_params
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
        required=None,
    )
    val_scores = run(params)
    print("\nFinished training GraphDRP model.")


if __name__ == "__main__":
    # main()
    main(sys.argv[1:])
