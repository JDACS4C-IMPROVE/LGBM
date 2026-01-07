""" Preprocess benchmark data (e.g., CSA data) to generate datasets for the
LightGBM prediction model.

Required outputs
----------------
All the outputs from this preprocessing script are saved in params["output_dir"].

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
from pathlib import Path
from typing import Dict

import pandas as pd
import joblib

# [Req] IMPROVE imports
# Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
# Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp

# Model-specifc imports
from model_params_def import preprocess_params # [Req]
from model_utils.utils import gene_selection, scale_df

filepath = Path(__file__).resolve().parent # [Req]


# [Req]
def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed (generated)
            ML data files.
    """
    # breakpoint()
    # from pprint import pprint; pprint(params);

    # ------------------------------------------------------
    # [Req] Load X data (feature representations)
    # ------------------------------------------------------
    # Use the provided data loaders to load data required by the model.
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
    print("\nLoads omics data.")
    omics_obj = omics_utils.OmicsLoader(params)
    ge = omics_obj.dfs['cancer_gene_expression.tsv'] # return gene expression

    print("\nLoad drugs data.")
    drugs_obj = drugs_utils.DrugsLoader(params)
    md = drugs_obj.dfs['drug_mordred.tsv'] # return the Mordred descriptors
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
    print("Create feature scaler.")
    rsp_tr = drp.DrugResponseLoader(params,
                                    split_file=params["train_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(params,
                                    split_file=params["val_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp = pd.concat([rsp_tr, rsp_vl], axis=0)

    # Retian feature rows that are present in the y data (response dataframe)
    # Intersection of omics features, drug features, and responses
    rsp = rsp.merge(ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
    rsp = rsp.merge(md[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
    ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
    md_sub = md[md[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)

    # Scale gene expression
    _, ge_scaler = scale_df(ge_sub, scaler_name=params["scaling"])
    ge_scaler_fpath = Path(params["output_dir"]) / params["ge_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)

    # Scale Mordred descriptors
    _, md_scaler = scale_df(md_sub, scaler_name=params["scaling"])
    md_scaler_fpath = Path(params["output_dir"]) / params["md_scaler_fname"]
    joblib.dump(md_scaler, md_scaler_fpath)
    print("Scaler object for Mordred:         ", md_scaler_fpath)

    del rsp, rsp_tr, rsp_vl, ge_sub, md_sub

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # All models must load response data (y data) using DrugResponseLoader().
    # Below, we iterate over the 3 split files (train, val, test) and load
    # response data, filtered by the split ids from the split files.

    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():

        # --------------------------------
        # [Req] Load response data
        # --------------------------------
        rsp = drp.DrugResponseLoader(params,
                                     split_file=split_file,
                                     verbose=False).dfs["response.tsv"]

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
        # [Req] Save ML data files in params["output_dir"]
        # The implementation of this step depends on the model.
        # --------------------------------
        # [Req] Build data name
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)

        print("Merge data")
        data = rsp.merge(ge_sc, on=params["canc_col_name"], how="inner")
        data = data.merge(md_sc, on=params["drug_col_name"], how="inner")
        data = data.sample(frac=1.0).reset_index(drop=True) # shuffle

        print("Save data")
        data = data.drop(columns=["study"]) # to_parquet() throws error since "study" contain mixed values
        data.to_parquet(Path(params["output_dir"]) / data_fname) # saves ML data file to parquet

        # Prepare the y dataframe for the current stage
        fea_list = ["ge", "mordred"]
        fea_cols = [c for c in data.columns if (c.split(fea_sep)[0]) in fea_list]
        meta_cols = [c for c in data.columns if (c.split(fea_sep)[0]) not in fea_list]
        ydf = data[meta_cols]

        # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(ydf, stage, params["output_dir"])

    return params["output_dir"]


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="lgbm_params.txt",
        additional_definitions=additional_definitions
    )
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
