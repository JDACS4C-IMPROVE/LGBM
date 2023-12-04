""" Preprocessing of raw data to generate datasets for GraphDRP Model. """

import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

# IMPROVE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

# Model-specific imports
# from model_utils.torch_utils import TestbedDataset

filepath = Path(__file__).resolve().parent # [Req]

# [Req] App-specific params (App: monotherapy drug response prediction)
# TODO: consider moving this list to drug_resp_pred.py module
# The list app_preproc_params must be copied here as is.
# The [Req] indicates that the args must be specified for the model.
app_preproc_params = [
    # These arg should be specified in the [modelname]_default_model.txt:
    # y_data_files, x_data_canc_files, x_data_drug_files
    {"name": "y_data_files", # default
     "type": str,
     "help": "List of files that contain the y (prediction variable) data. \
             Example: [['response.tsv']]",
    },
    {"name": "x_data_canc_files", # [Req]
     "type": str,
     "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    },
    {"name": "x_data_drug_files", # [Req]
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

# [Optional] Model-specific params (Model: LightGBM)
# All args in model_preproc_params are optional. If no args are required by the model, then it should be an empty list.
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

# [Req]
preprocess_params = app_preproc_params + model_preproc_params
req_preprocess_args = [ll["name"] for ll in preprocess_params]

# ------------------------------------------------------------
def gene_selection(df, genes_fpath, canc_col_name):
    """ Takes a dataframe omics data (e.g., gene expression) and retains only
    the genes specified in genes_fpath.
    """
    with open(genes_fpath) as f:
        genes = [str(line.rstrip()) for line in f]
    # genes = ["ge_" + str(g) for g in genes]  # This is for our legacy data
    # print("Genes count: {}".format(len(set(genes).intersection(set(df.columns[1:])))))
    # genes = list(set(genes).intersection(set(df.columns[1:])))
    genes = drp.common_elements(genes, df.columns[1:])
    cols = [canc_col_name] + genes
    return df[cols]


def scale_df(df, scaler_name: str="std", scaler=None, verbose: bool=False):
    """ Returns a dataframe with scaled data.

    It can create a new scaler or use the scaler passed or return the
    data as it is. If `scaler_name` is None, no scaling is applied. If
    `scaler` is None, a new scaler is constructed. If `scaler` is not
    None, and `scaler_name` is not None, the scaler passed is used for
    scaling the data frame.

    Args:
        df: Pandas dataframe to scale.
        scaler_name: Name of scikit learn scaler to apply. Options:
                     ["minabs", "minmax", "std", "none"]. Default: std
                     standard scaling.
        scaler: Scikit object to use, in case it was created already.
                Default: None, create scikit scaling object of
                specified type.
        verbose: Flag specifying if verbose message printing is desired.
                 Default: False, no verbose print.

    Returns:
        pd.Dataframe: dataframe that contains drug response values.
        scaler: Scikit object used for scaling.
    """
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return df, None

    # Scale data
    # Select only numerical columns in data frame
    df_num = df.select_dtypes(include="number")

    if scaler is None: # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "minabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            print(f"The specified scaler ({scaler_name}) is not implemented (no df scaling).")
            return df, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else: # Apply passed scikit scaler
        # Scale data according to specified scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    df[df_num.columns] = df_norm
    return df, scaler
# ------------------------------------------------------------


def compose_data_arrays(df_response: pd.DataFrame,
                        df_drug: pd.DataFrame,
                        df_cell: pd.DataFrame,
                        drug_col_name: str,
                        canc_col_name: str):
    """ Returns drug and cancer feature data, and response values.

    :params: pd.Dataframe df_response: drug response dataframe. This
             already has been filtered to three columns: drug_id,
             cell_id and drug_response.
    :params: pd.Dataframe df_drug: drug features dataframe.
    :params: pd.Dataframe df_cell: cell features dataframe.
    :params: str drug_col_name: Column name that contains the drug ids.
    :params: str canc_col_name: Column name that contains the cancer sample ids.

    :return: Numpy arrays with drug features, cell features and responses
            xd, xc, y
    :rtype: np.array
    """
    xd = [] # To collect drug features
    xc = [] # To collect cell features
    y = []  # To collect responses

    # To collect missing or corrupted data
    nan_rsp_list = []
    miss_cell = []
    miss_drug = []
    # count_nan_rsp = 0
    # count_miss_cell = 0
    # count_miss_drug = 0

    # Convert to indices for rapid lookup (??)
    df_drug = df_drug.set_index([drug_col_name])
    df_cell = df_cell.set_index([canc_col_name])

    for i in range(df_response.shape[0]):  # tuples of (drug name, cell id, response)
        if i > 0 and (i%15000 == 0):
            print(i)
        drug, cell, rsp = df_response.iloc[i, :].values.tolist()
        if np.isnan(rsp):
            # count_nan_rsp += 1
            nan_rsp_list.append(i)
        # If drug and cell features are available
        try: # Look for drug
            drug_features = df_drug.loc[drug]
        except KeyError: # drug not found
            miss_drug.append(drug)
            # count_miss_drug += 1
        else: # Look for cell
            try:
                cell_features = df_cell.loc[cell]
            except KeyError: # cell not found
                miss_cell.append(cell)
                # count_miss_cell += 1
            else: # Both drug and cell were found
                xd.append(drug_features.values) # xd contains list of drug feature vectors
                xc.append(cell_features.values) # xc contains list of cell feature vectors
                y.append(rsp)

    # print("Number of NaN responses:   ", len(nan_rsp_list))
    # print("Number of drugs not found: ", len(miss_cell))
    # print("Number of cells not found: ", len(miss_drug))

    # # Reset index
    # df_drug = df_drug.reset_index()
    # df_cell = df_cell.reset_index()

    return np.asarray(xd).squeeze(), np.asarray(xc), np.asarray(y)


def run(params: Dict):
    """ Run data pre-processing for GraphDRP model.
    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Build paths and create ML data dir
    # ----------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)  

    # Create outdir for ML data (to save preprocessed data)
    # processed_outdir = frm.create_ml_data_outdir(params)  # creates params["ml_data_outdir"]
    frm.create_outdir(outdir=params["ml_data_outdir"])
    # ----------------------------------------

    # ------------------------------------------------------
    # [Req] Load omics data
    # ---------------------
    print("\nLoading omics data ...")
    oo = drp.OmicsLoader(params)
    # print(oo)
    ge = oo.dfs['cancer_gene_expression.tsv']  # get only gene expression dataframe
    # ---------------------

    # ------------------------------------------------------
    # [Optional] Prep omics data
    # --------------------------
    # Gene selection (LINCS landmark genes) for GraphDRP
    if params["use_lincs"]:
        genes_fpath = filepath/"landmark_genes"
        ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

    fea_sep = "."
    ge = ge.rename(columns={c: f"ge{fea_sep}{c}" for c in ge.columns[1:]})
    # --------------------------

    # ------------------------------------------------------
    # [Req] Load drug data
    # --------------------
    print("\nLoading drugs data...")
    dd = drp.DrugsLoader(params)
    # print(dd)
    md = dd.dfs['drug_mordred.tsv']  # get only the Mordred data
    md = md.reset_index()
    # --------------------

    # -------------------------------------------
    # Construct ML data for every stage (train, val, test)
    # [Req] All models must load response data (y data) using DrugResponseLoader().
    # Below, we iterate over the 3 split files (train, val, test) and load response
    # data, filtered by the split ids from the split files.
    # -------------------------------------------
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}
    scaler = None

    # %%%%%%%%%%%%%%%
    # Create scaler
    # %%%%%%%%%%%%%%%
    # Combine responses
    # import ipdb; ipdb.set_trace()
    rsp_tr = drp.DrugResponseLoader(params,
                                    split_file=params["train_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(params,
                                    split_file=params["val_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp_te = drp.DrugResponseLoader(params,
                                    split_file=params["test_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp_dev = pd.concat([rsp_tr, rsp_vl, rsp_te], axis=0)
    print(rsp_tr.shape) 
    print(rsp_vl.shape) 
    print(rsp_te.shape) 
    print(rsp_dev.shape) 

    # # Intersection of drugs, cells, and responses
    # # import ipdb; ipdb.set_trace()
    # rsp_dev_sub, ge_sub = drp.get_common_samples(df1=rsp_dev, df2=ge,
    #                                              ref_col=params["canc_col_name"])
    # rsp_dev_sub, md_sub = drp.get_common_samples(df1=rsp_dev_sub, df2=md,
    #                                              ref_col=params["drug_col_name"])

    # Intersection of drugs, cells, and responses
    # import ipdb; ipdb.set_trace()
    df = rsp_dev.merge(ge[params["canc_col_name"]],
                       on=params["canc_col_name"], how="inner")
    df = df.merge(md[params["drug_col_name"]],
                  on=params["drug_col_name"], how="inner")
    ge_sub = ge[ge[params["canc_col_name"]].isin(df[params["canc_col_name"]])].reset_index(drop=True)
    md_sub = md[md[params["drug_col_name"]].isin(df[params["drug_col_name"]])].reset_index(drop=True)
    rsp_dev_sub = df
    print(rsp_dev_sub.shape) 
    print(ge_sub.shape) 
    print(md_sub.shape) 

    # Scale
    # import ipdb; ipdb.set_trace()
    ge_sc, ge_scaler = scale_df(ge_sub, scaler_name=params["scaling"])
    md_sc, md_scaler = scale_df(md_sub, scaler_name=params["scaling"])
    ge_scaler_fpath = Path(params["ml_data_outdir"]) / params["ge_scaler_fname"]
    md_scaler_fpath = Path(params["ml_data_outdir"]) / params["md_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    joblib.dump(md_scaler, md_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)
    print("Scaler object for Mordred:         ", md_scaler_fpath)
    del rsp_tr, rsp_vl, rsp_te, rsp_dev, rsp_dev_sub, ge_sub, md_sub, ge_sc, md_sc
    # %%%%%%%%%%%%%%%

    # import ipdb; ipdb.set_trace()
    for stage, split_file in stages.items():

        # ------------------------
        # [Req] Load response data
        # ------------------------
        # rr = drp.DrugResponseLoader(params, split_file=split_file, verbose=False)
        # df_response = rr.dfs["response.tsv"]
        df_rsp = drp.DrugResponseLoader(params,
                                        split_file=split_file,
                                        verbose=False).dfs["response.tsv"]
        # ------------------------

        # ------------------------
        # [GraphDRP] Data prep
        # ------------------------
        # Retain (canc, drug) response samples for which omics data is available
        # import ipdb; ipdb.set_trace()
        df = df_rsp.merge(ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
        df = df.merge(md[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
        ge_sub = ge[ge[params["canc_col_name"]].isin(df[params["canc_col_name"]])].reset_index(drop=True)
        md_sub = md[md[params["drug_col_name"]].isin(df[params["drug_col_name"]])].reset_index(drop=True)
        # gg = ge_sub.copy(); mm = md_sub.copy()

        # ydf, ge_sub = drp.get_common_samples(df1=df_rsp, df2=ge, ref_col=params["canc_col_name"])
        # ydf, md_sub = drp.get_common_samples(df1=ydf, df2=md, ref_col=params["drug_col_name"])

        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler)  # Use dev scaler!
        md_sc, _ = scale_df(md_sub, scaler=md_scaler)  # Use dev scaler!
        print("GE mean:", ge_sc.iloc[:,1:].mean(axis=0).mean())
        print("GE var: ", ge_sc.iloc[:,1:].var(axis=0).mean())
        print("MD mean:", md_sc.iloc[:,1:].mean(axis=0).mean())
        print("MD var: ", md_sc.iloc[:,1:].var(axis=0).mean())
        # ------------------------

        # -----------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step, depends on the model.
        # -----------------------
        # import ipdb; ipdb.set_trace()
        # # [Req] Save y dataframe for the current stage
        # frm.save_stage_ydf(ydf, params, stage)

        # [Req] Create data name
        # data_fname = frm.build_ml_data_name(params, stage,
        #                                     file_format=params["data_format"])
        data_fname = frm.build_ml_data_name(params, stage)

        # Revmoe data_format because TestbedDataset() appends '.pt' to the
        # file name automatically. This is unique for GraphDRP.
        # data_fname = data_fname.split(params["data_format"])[0]

        # Create the ml data and save it as data_fname in params["ml_data_outdir"]
        # Note! In the *train*.py and *infer*.py scripts, functionality should
        # be implemented to load the saved data.
        # -----
        # In GraphDRP, TestbedDataset() is used to create and save the file.
        # TestbedDataset() which inherits from torch_geometric.data.InMemoryDataset
        # automatically creates dir called "processed" inside root and saves the file
        # inside. This results in: [root]/processed/[dataset],
        # e.g., ml_data/processed/train_data.pt
        # -----
        # TestbedDataset(root=params["ml_data_outdir"],
        #                dataset=data_fname,
        #                xd=xd,
        #                xt=xc,
        #                y=y,
        #                smile_graph=smiles_graphs)

        print("Merge data")
        data = pd.merge(df, ge_sc, on=params["canc_col_name"], how="inner")
        data = pd.merge(data, md_sc, on=params["drug_col_name"], how="inner")  # mess up y data
        data = data.sample(frac=1.0).reset_index(drop=True)

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


# def main():
def main(args):
    # import ipdb; ipdb.set_trace()
    # [Req]
    params = frm.initialize_parameters(
        filepath,
        # default_model="graphdrp_default_model.txt",
        # default_model="params_ws.txt",
        # default_model="params_cs.txt",
        default_model="lgbm_params.txt",
        # default_model="lgb_params_ws.txt",
        # default_model="lgb_params_cs.txt",
        additional_definitions=preprocess_params,
        required=req_preprocess_args,
    )
    ml_data_outdir = run(params)
    print("\nFinished GraphDRP pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    # main()
    main(sys.argv[1:])
