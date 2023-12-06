"""  Preprocess benchmark data (e.g., CSA data) to generate datasets for the
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
from pathlib import Path
from typing import Dict

import pandas as pd
import joblib

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

# Model-specifc imports
from model_utils.utils import gene_selection, scale_df

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
# required:  these params must be specified for the model
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

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
preprocess_params = app_preproc_params + model_preproc_params
# req_preprocess_params = []
# ---------------------


# [Req]
def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed (generated)
            ML data files.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Build paths and create ML data dir
    # ------------------------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)  

    # Create output dir for ML data (to save preprocessed data)
    frm.create_outdir(outdir=params["ml_data_outdir"])

    # ------------------------------------------------------
    # [Req] Load X data (feature representations)
    # ------------------------------------------------------
    # Benchmark data includes three dirs: x_data, y_data, and splits.
    # The x_data contains files that represent feature information such as
    # cancer representation (omics data) and drug representation (SMILES, etc.).
    # Prediction models utilize various types of feature representations.
    # Drug response prediction (DRP) models generally use omics and drug features.
    # ...
    # If the model uses omics data types that are provided as part of benchmark
    # data, then the model must use the provided data loaders to load the data files
    # from the x_data dir.
    print("\nLoads omics data.")
    omics_obj = drp.OmicsLoader(params)
    # print(omics_obj)
    ge = omics_obj.dfs['cancer_gene_expression.tsv'] # return gene expression

    print("\nLoad drugs data.")
    drugs_obj = drp.DrugsLoader(params)
    # print(drugs_obj)
    md = drugs_obj.dfs['drug_mordred.tsv'] # return the Mordred descriptors
    md = md.reset_index()

    # ------------------------------------------------------
    # [Optional] Further preprocess X data
    # ------------------------------------------------------
    # Gene selection (LINCS landmark genes)
    if params["use_lincs"]:
        genes_fpath = filepath/"landmark_genes"
        ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

    # Prefix gene column names with "ge."
    fea_sep = "."
    fea_prefix = "ge"
    ge = ge.rename(columns={fea: f"{fea_prefix}{fea_sep}{fea}" for fea in ge.columns[1:]})

    # ------------------------------------------------------
    # [Optional] Create feature scaler
    # ------------------------------------------------------
    # Load and combine responses
    rsp_tr = drp.DrugResponseLoader(params,
                                    split_file=params["train_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(params,
                                    split_file=params["val_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp_te = drp.DrugResponseLoader(params,
                                    split_file=params["test_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp = pd.concat([rsp_tr, rsp_vl, rsp_te], axis=0)
    # print(rsp_tr.shape) 
    # print(rsp_vl.shape) 
    # print(rsp_te.shape) 
    # print(rsp.shape) 

    # # Intersection of drugs, cells, and responses
    # # import ipdb; ipdb.set_trace()
    # rsp_dev_sub, ge_sub = drp.get_common_samples(df1=rsp_dev, df2=ge,
    #                                              ref_col=params["canc_col_name"])
    # rsp_dev_sub, md_sub = drp.get_common_samples(df1=rsp_dev_sub, df2=md,
    #                                              ref_col=params["drug_col_name"])

    # Intersection of omics features, drug features, and responses
    df = rsp.merge(ge[params["canc_col_name"]],
                       on=params["canc_col_name"], how="inner")
    df = df.merge(md[params["drug_col_name"]],
                  on=params["drug_col_name"], how="inner")
    ge_sub = ge[ge[params["canc_col_name"]].isin(df[params["canc_col_name"]])].reset_index(drop=True)
    md_sub = md[md[params["drug_col_name"]].isin(df[params["drug_col_name"]])].reset_index(drop=True)
    # rsp_sub = df
    # print(rsp_sub.shape)
    # print(ge_sub.shape)
    # print(md_sub.shape)

    # Scale
    ge_sc, ge_scaler = scale_df(ge_sub, scaler_name=params["scaling"])
    md_sc, md_scaler = scale_df(md_sub, scaler_name=params["scaling"])
    ge_scaler_fpath = Path(params["ml_data_outdir"]) / params["ge_scaler_fname"]
    md_scaler_fpath = Path(params["ml_data_outdir"]) / params["md_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    joblib.dump(md_scaler, md_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)
    print("Scaler object for Mordred:         ", md_scaler_fpath)
    del rsp, rsp_tr, rsp_vl, rsp_te, df, ge_sub, md_sub, ge_sc, md_sc

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # All models must load response data (y data) using DrugResponseLoader().
    # Below, we iterate over the 3 split files (train, val, test) and load
    # response data, filtered by the split ids from the split files.
    # ------------------------------------------------------
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}
    scaler = None

    # # %%%%%%%%%%%%%%%
    # # Create scaler
    # # %%%%%%%%%%%%%%%
    # # Combine responses
    # # import ipdb; ipdb.set_trace()
    # rsp_tr = drp.DrugResponseLoader(params,
    #                                 split_file=params["train_split_file"],
    #                                 verbose=False).dfs["response.tsv"]
    # rsp_vl = drp.DrugResponseLoader(params,
    #                                 split_file=params["val_split_file"],
    #                                 verbose=False).dfs["response.tsv"]
    # rsp_te = drp.DrugResponseLoader(params,
    #                                 split_file=params["test_split_file"],
    #                                 verbose=False).dfs["response.tsv"]
    # rsp_dev = pd.concat([rsp_tr, rsp_vl, rsp_te], axis=0)
    # print(rsp_tr.shape) 
    # print(rsp_vl.shape) 
    # print(rsp_te.shape) 
    # print(rsp_dev.shape) 

    # # # Intersection of drugs, cells, and responses
    # # # import ipdb; ipdb.set_trace()
    # # rsp_dev_sub, ge_sub = drp.get_common_samples(df1=rsp_dev, df2=ge,
    # #                                              ref_col=params["canc_col_name"])
    # # rsp_dev_sub, md_sub = drp.get_common_samples(df1=rsp_dev_sub, df2=md,
    # #                                              ref_col=params["drug_col_name"])

    # # Intersection of drugs, cells, and responses
    # # import ipdb; ipdb.set_trace()
    # df = rsp_dev.merge(ge[params["canc_col_name"]],
    #                    on=params["canc_col_name"], how="inner")
    # df = df.merge(md[params["drug_col_name"]],
    #               on=params["drug_col_name"], how="inner")
    # ge_sub = ge[ge[params["canc_col_name"]].isin(df[params["canc_col_name"]])].reset_index(drop=True)
    # md_sub = md[md[params["drug_col_name"]].isin(df[params["drug_col_name"]])].reset_index(drop=True)
    # rsp_dev_sub = df
    # print(rsp_dev_sub.shape) 
    # print(ge_sub.shape) 
    # print(md_sub.shape) 

    # # Scale
    # # import ipdb; ipdb.set_trace()
    # ge_sc, ge_scaler = scale_df(ge_sub, scaler_name=params["scaling"])
    # md_sc, md_scaler = scale_df(md_sub, scaler_name=params["scaling"])
    # ge_scaler_fpath = Path(params["ml_data_outdir"]) / params["ge_scaler_fname"]
    # md_scaler_fpath = Path(params["ml_data_outdir"]) / params["md_scaler_fname"]
    # joblib.dump(ge_scaler, ge_scaler_fpath)
    # joblib.dump(md_scaler, md_scaler_fpath)
    # print("Scaler object for gene expression: ", ge_scaler_fpath)
    # print("Scaler object for Mordred:         ", md_scaler_fpath)
    # del rsp_tr, rsp_vl, rsp_te, rsp_dev, rsp_dev_sub, ge_sub, md_sub, ge_sc, md_sc
    # # %%%%%%%%%%%%%%%

    for stage, split_file in stages.items():

        # --------------------------------
        # [Req] Load response data
        # --------------------------------
        rsp = drp.DrugResponseLoader(params,
                                     split_file=split_file,
                                     verbose=False).dfs["response.tsv"]

        # --------------------------------
        # [Optional] Data prep
        # --------------------------------
        # Retain (canc, drug) responses for which both omics and drug features
        # are available.
        rsp = rsp.merge(ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
        rsp = rsp.merge(md[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
        ge_sub = ge[ge[params["canc_col_name"]].isin(
            rsp[params["canc_col_name"]])].reset_index(drop=True)
        md_sub = md[md[params["drug_col_name"]].isin(
            rsp[params["drug_col_name"]])].reset_index(drop=True)

        # ydf, ge_sub = drp.get_common_samples(df1=rsp, df2=ge, ref_col=params["canc_col_name"])
        # ydf, md_sub = drp.get_common_samples(df1=ydf, df2=md, ref_col=params["drug_col_name"])

        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler)  # Use dev scaler!
        md_sc, _ = scale_df(md_sub, scaler=md_scaler)  # Use dev scaler!
        print("GE mean:", ge_sc.iloc[:,1:].mean(axis=0).mean())
        print("GE var: ", ge_sc.iloc[:,1:].var(axis=0).mean())
        print("MD mean:", md_sc.iloc[:,1:].mean(axis=0).mean())
        print("MD var: ", md_sc.iloc[:,1:].var(axis=0).mean())

        # --------------------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step, depends on the model.
        # --------------------------------
        # [Req] Build data name
        data_fname = frm.build_ml_data_name(params, stage)

        print("Merge data")
        data = pd.merge(rsp, ge_sc, on=params["canc_col_name"], how="inner")
        data = pd.merge(data, md_sc, on=params["drug_col_name"], how="inner")
        data = data.sample(frac=1.0).reset_index(drop=True) # shuffle

        print("Save data")
        # data.to_csv(Path(params["ml_data_outdir"])/data_fname, index=False)
        data = data.drop(columns=["study"])
        data.to_parquet(Path(params["ml_data_outdir"])/data_fname)

        # [Req] Save y dataframe for the current stage
        fea_list = ["ge", "mordred"]
        fea_cols = [c for c in data.columns if (c.split(fea_sep)[0]) in fea_list]
        meta_cols = [c for c in data.columns if (c.split(fea_sep)[0]) not in fea_list]
        ydf = data[meta_cols]
        frm.save_stage_ydf(ydf, params, stage)

    return params["ml_data_outdir"]


# [Req]
# def main():
def main(args):
    # [Req]
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        default_model="lgbm_params.txt",
        # default_model="lgbm_params_ws.txt",
        # default_model="lgbm_params_cs.txt",
        additional_definitions=additional_definitions,
        # required=req_preprocess_params,
        required=None,
    )
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    # main()
    main(sys.argv[1:])
