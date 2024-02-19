""" Preprocess benchmark data (e.g., CSA data) to generate datasets for the
LightGBM prediction model.

Required outputs
----------------
All the outputs from this preprocessing script are saved in params["ml_data_outdir"].

1. Model input data files.
   This script creates three data files corresponding to train, validation,
   and test data. These data files are used as inputs to the ML/DL model in
   the train and infer scripts. The file format is specified by
   params["data_format"].
   For LightGBM, the generated files:
        train_data.csv, val_data.csv, test_data.csv

2. Y data files.
   The script creates dataframes with true y values and additional metadata.
   Generated files:
        train_y_data.csv, val_y_data.csv, and test_y_data.csv.
"""

import sys
import os
from pathlib import Path
from typing import Dict
import logging

import pandas as pd
import joblib

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm 
from improve import drug_resp_pred as drp

from improve import config as BaseConfig
from improve import preprocess as PreprocessConfig
from improve.Benchmarks.DrugResponsePrediction import DRP as BenchmanrkDRP

# Model-specifc imports
from model_utils.utils import gene_selection, scale_df

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL" , logging.DEBUG))


filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_preproc_params
# 2. model_preproc_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Note! This list should not be modified (i.e., no params should added or
# removed from the list.
# 
# There are two types of params in the list: default and required
# default:   default values should be used
# required:  these params must be specified for the model in the param file
app_preproc_params = [
    {"name": "y_data_files", # default
     "type": str,
     "help": "List of files that contain the y (prediction variable) data. \
             Example: [['response.tsv']]",
    },
    {"name": "x_data_canc_files", # required
     "type": str,
     "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    },
    {"name": "x_data_drug_files", # required
     "type": str,
     "help": "List of feature files. Examples: \n\
             1) [['drug_SMILES.tsv']] \n\
             2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
    },
    {"name": "canc_col_name",
     "default": "improve_sample_id", # default
     "type": str,
     "help": "Column name in the y (response) data file that contains the cancer sample ids.",
    },
    {"name": "drug_col_name", # default
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name in the y (response) data file that contains the drug ids.",
    },
]

# 2. Model-specific params (Model: LightGBM)
# All params in model_preproc_params are optional.
# If no params are required by the model, then it should be an empty list.
model_preproc_params = [
    {"name": "use_lincs",
     "type": frm.str2bool,
     "default": True,
     "help": "Flag to indicate if landmark genes are used for gene selection.",
    },
    {"name": "scaling",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "miabs", "robust"],
     "help": "Scaler for gene expression and Mordred descriptors data.",
    },
    {"name": "ge_scaler_fname",
     "type": str,
     "default": "x_data_gene_expression_scaler.gz",
     "help": "File name to save the gene expression scaler object.",
    },
    {"name": "md_scaler_fname",
     "type": str,
     "default": "x_data_mordred_scaler.gz",
     "help": "File name to save the Mordred scaler object.",
    },
]

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
# preprocess_params = app_preproc_params + model_preproc_params
preprocess_params = model_preproc_params
# ---------------------


# [Req]
def run(cfg: PreprocessConfig.Preprocess):
    """ Run data preprocessing.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed (generated)
            ML data files.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Build paths and create output dir
    # ------------------------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
  
    try:
        if cfg.get_param("subparser_name") is None or cfg.get_param("subparser_name") == "":
            logger.error("Subparser name is not set.")
            # throw error
            raise ValueError("Missing mandatory positional parameter: subparser_name.")
        else:
            logger.info(f"Subparser name: {cfg.get_param('subparser_name')}")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    
    # frm.create_outdir(outdir=params["ml_data_outdir"])

    # ------------------------------------------------------
    # [Req] Load X data (feature representations)
    # ------------------------------------------------------
    # Use the provided data loaders to load data that is required by the model.
    #
    # Benchmark data includes three dirs: x_data, y_data, splits.
    # The x_data contains files that represent feature information such as
    # cancer representation (e.g., omics) and drug representation (e.g., SMILES).
    #
    # Prediction models utilize various types of feature representations.
    # Drug response prediction (DRP) models generally use omics and drug features.
    #
    # If the model uses omics data types that are provided as part of the benchmark
    # data, then the model must use the provided data loaders to load the data files
    # from the x_data dir.
        
    logger.debug("Load Data from Preprocess")
    cfg.load_data()
    # print("\nLoads omics data.")

    logger.debug(f"x_data_canc_files: {cfg.get_param('x_data_canc_files')}")
    logger.debug(f"x_data_canc_files: {cfg.dict()['x_data_canc_files']}")

    params = cfg.dict()
    # params['x_data_path'] = DRP.x_data_path
    # params['y_data_path'] = DRP.y_data_path
    # params['splits_path'] = DRP.splits_path
    logger.debug(f"params: {params['x_data_path']}")
    sys.exit(1)

    # logger.debug("Loading from Alex's code")
    # omics_obj = drp.OmicsLoader(params)

    logger.debug("Loading from DRP class")
    DRP.load_data(verbose=True)



    # print(omics_obj)
    ge = DRP.omics.dfs['cancer_gene_expression.tsv'] # return gene expression

    # print("\nLoad drugs data.")
    # drugs_obj = drp.DrugsLoader(params)
    # print(drugs_obj)
    md = DRP.drugs.dfs['drug_mordred.tsv'] # return the Mordred descriptors
    md = md.reset_index()  # TODO. implement reset_index() inside the loader

    # ------------------------------------------------------
    # Further preprocess X data
    # ------------------------------------------------------
    # Gene selection (based on LINCS landmark genes)
    if params["use_lincs"]:
        genes_fpath = filepath/"model_utils/landmark_genes.txt"
        ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

    # Prefix gene column names with "ge."
    fea_sep = "."
    fea_prefix = "ge"
    ge = ge.rename(columns={fea: f"{fea_prefix}{fea_sep}{fea}" for fea in ge.columns[1:]})

    # ------------------------------------------------------
    # Create feature scaler
    # ------------------------------------------------------
    # Load and combine responses
    logger.info("Create feature scaler.")
    # rsp_tr = drp.DrugResponseLoader(params,
    #                                 split_file=params["train_split_file"],
    #                                 verbose=False).dfs["response.tsv"]
    # rsp_vl = drp.DrugResponseLoader(params,
    #                                 split_file=params["val_split_file"],
    #                                 verbose=False).dfs["response.tsv"]
    rsp = pd.concat([DRP.train, DRP.validate], axis=0)

    # Retian feature rows that are present in the y data (response dataframe)
    # Intersection of omics features, drug features, and responses
    rsp = rsp.merge(ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
    rsp = rsp.merge(md[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
    ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
    md_sub = md[md[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)

    # Scale gene expression
    _, ge_scaler = scale_df(ge_sub, scaler_name=params["scaling"])
    ge_scaler_fpath = Path(params['output_dir']) / params["ge_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)

    # Scale Mordred descriptors
    _, md_scaler = scale_df(md_sub, scaler_name=params["scaling"])
    md_scaler_fpath = Path(params["output_dir"]) / params["md_scaler_fname"]
    joblib.dump(md_scaler, md_scaler_fpath)
    print("Scaler object for Mordred:         ", md_scaler_fpath)

    del rsp, ge_sub, md_sub

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # All models must load response data (y data) using DrugResponseLoader().
    # Below, we iterate over the 3 split files (train, val, test) and load
    # response data, filtered by the split ids from the split files.

    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {"train": params["train_split_file"],
              "validate": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():

        logger.debug(f"Stage: {stage}, Split file: {split_file}")
        # --------------------------------
        # [Req] Load response data
        # --------------------------------
        # rsp = drp.DrugResponseLoader(params,
        #                              split_file=split_file,
        #                              verbose=False).dfs["response.tsv"]

        rsp =getattr(DRP, stage , None)
        print(rsp)

        # --------------------------------
        # Data prep
        # --------------------------------
        # Retain (canc, drug) responses for which both omics and drug features
        # are available.
        rsp = rsp.merge(ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
        rsp = rsp.merge(md[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
        ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
        md_sub = md[md[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)

        # Scale features
        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler) # scale gene expression
        md_sc, _ = scale_df(md_sub, scaler=md_scaler) # scale Mordred descriptors
        # print("GE mean:", ge_sc.iloc[:,1:].mean(axis=0).mean())
        # print("GE var: ", ge_sc.iloc[:,1:].var(axis=0).mean())
        # print("MD mean:", md_sc.iloc[:,1:].mean(axis=0).mean())
        # print("MD var: ", md_sc.iloc[:,1:].var(axis=0).mean())

        # --------------------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step depends on the model.
        # --------------------------------
        # [Req] Build data name
        data_fname = frm.build_ml_data_name(params, stage)

        print("Merge data")
        data = rsp.merge(ge_sc, on=params["canc_col_name"], how="inner")
        data = data.merge(md_sc, on=params["drug_col_name"], how="inner")
        data = data.sample(frac=1.0).reset_index(drop=True) # shuffle

        print("Save data")
        data = data.drop(columns=["study"]) # to_parquet() throws error since "study" contain mixed values
        data.to_parquet(Path(params["output_dir"])/data_fname) # saves ML data file to parquet

        # Prepare the y dataframe for the current stage
        fea_list = ["ge", "mordred"]
        fea_cols = [c for c in data.columns if (c.split(fea_sep)[0]) in fea_list]
        meta_cols = [c for c in data.columns if (c.split(fea_sep)[0]) not in fea_list]
        ydf = data[meta_cols]

        # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(ydf, params, stage)

    return params["output_dir"]


# [Req]
def main(args):
    """ Main function to run data preprocessing."""

    # Additional definitions
    additional_definitions = preprocess_params

    # Initialize Config and CLI
    logger.info("Initialize Config and CLI.")
    preprocess_config = PreprocessConfig.Preprocess()

    logger.debug("Initialize parameters.")
    params = preprocess_config.initialize_parameters(
        filepath,
        default_model="lgbm_params.txt",
        default_config="lgbm.cfg",
        # default_model="params_ws.txt",
        # default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        required=None,
    )


    logger.info("Run data preprocessing.")
    ml_data_outdir = run(preprocess_config)
    logger.info(f"Preprocessed ML data is saved in {preprocess_config.output_dir}")



# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
