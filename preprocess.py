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
from pathlib import Path
from typing import Dict

import pandas as pd
import joblib

# [Req] IMPROVE/CANDLE imports
from improvelib import framework as frm
from improvelib import drug_resp_pred as drp

from improvelib import config as BaseConfig

# Model-specifc imports
from model_utils.utils import gene_selection, scale_df
import logging
import os

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
preprocess_params = app_preproc_params + model_preproc_params
# ---------------------


class Preprocess(BaseConfig.Config):
    """Class to handle configuration files for Preprocessing."""

    # Set section for config file
    section = 'Preprocess'

    # Set options for command line
    preprocess_options = []

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger("Preprocess")
        self.logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL" , logging.INFO))
      
       
        self.options = Preprocess.preprocess_options
        # Set subparser for benchmark and file
        subparsers=self.cli.parser.add_subparsers(dest='subparser_name')
        # Benchmark subparser
        benchmark=subparsers.add_parser('benchmark', help='Use DRPBenchmark_v1.0')
        benchmark.add_argument('--benchmark_type', choices=['DRP', 'Default'], help='Specify benchmark format, e.g. DRP for DRPBenchmark_v1.0')
        benchmark.add_argument('--benchmark_dir', metavar='DIR', type=str, dest="benchmark_dir",
                                    default=os.getenv("IMPROVE_BENCHMARK_DIR" , "./"), 
                                    help='Base directory for DRPBenchmark_v1.0 data. Default is IMPROVE_BENCHMARK_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        
        drp=benchmark.add_argument_group('DRPBenchmark_v1.0', 'Options for drug response prediction benchmark v1.0')
        drp.add_argument('--drp', action='store_true', help='Use DRPBenchmark_v1.0')
        drp.add_argument('--drp_dir', metavar='DIR', type=str, dest="benchmark_dir",
                                    default=os.getenv("IMPROVE_BENCHMARK_DIR" , "./"), 
                                    help='Base directory for DRPBenchmark_v1.0 data. Default is IMPROVE_BENCHMARK_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        
        drp.add_argument("--dataset", type=str, default=None , help="Name of dataset")
        drp.add_argument("--split_id", type=str, default="y_data" , help="Split ID for the dataset")
        drp.add_argument("--splits_dir", type=str, default="splits" , help="Dir name that contains files that store split ids of the y data file.")
        drp.add_argument("--metric", type=str, default="auc" , 
                         help="Metric for drug response prediction problem it can be IC50, AUC, and others.")
        
        drp.add_argument("--feature_data_format", type=str, default="parquet" ,
                        help="Output format for the preprocessed data. Default is parquet.")
        drp.add_argument("--output_dir", type=str, default="auc" , 
                         help="Metric for drug response prediction problem it can be IC50, AUC, and others.")
        

    
    def get_param(self, key):
        """Get a parameter from the Preprocessing config."""
        return super().get_param(Preprocess.section, key)
    
    def set_params(self, key=None, value=None):
        print( "set_params" + type(self))
        return super().set_param(Preprocess.section, key, value)
    
    def set_param(self, key=None, value=None):
        return super().set_param(Preprocess.section, key, value)

    def dict(self):
        """Get the Preprocessing config as a dictionary."""
        return super().dict(Preprocess.section)

    def initialize_parameters(self, pathToModelDir, section='Preprocess', default_config='default.cfg', default_model=None, additional_definitions=None, required=None):
        """Initialize Command line Interfcace and config for Preprocessing."""
        self.logger.debug("Initializing parameters for Preprocessing.")
        print( "initialize_parameters" + str(type(self)) )

        if additional_definitions :
            self.options = self.options + additional_definitions
       
        p = super().initialize_parameters(pathToModelDir, section, default_config, default_model, self.options , required)
        print(self.get_param("log_level"))
        self.logger.setLevel(self.get_param("log_level"))
        return p




# [Req]
def run(cfg, params: Dict):
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

    # Create output dir for model input data (to save preprocessed ML data)
    params = drp.ParameterConverter().update_params(params)
    frm.create_outdir(outdir=str(params["output_dir"]))

    benchmark_dir = params["benchmark_dir"]
    benchmark = drp.SingleDRPBenchmark()
    benchmark.set_benchmark_dir(benchmark_dir)

    benchmark.set_dataset(params["dataset"])
    benchmark.set_split_id(params["split_id"])
    benchmark.set_splits_dir(params["splits_dir"])
    benchmark.set_drp_metric(params["metric"])

    ge = benchmark.get_full_dataframe(drp.SingleDRPDataFrame.CELL_LINE_GENE_EXPRESSION)
    md = benchmark.get_full_dataframe(drp.SingleDRPDataFrame.DRUG_MORDRED)
    md = md.reset_index()


    # Prefix gene column names with "ge."
    fea_sep = "."
    def update_gene_expression(ge, feature_separator):
        ge.columns = ge.columns.get_level_values('Gene_Symbol').values
    
        # HACKY STUFF, SHOULD BE ADDRESSED IN THE BENCHMARK DATAFILE
        ge.columns = [benchmark.CANCER_COL_NAME] + ge.columns.tolist()[1:]
        #
        if params["use_lincs"]:
            genes_fpath = filepath/"model_utils/landmark_genes.txt"
            ge = gene_selection(ge, genes_fpath, canc_col_name=benchmark.CANCER_COL_NAME)
        fea_prefix = "ge"
        ge = ge.rename(columns={fea: f"{fea_prefix}{fea_sep}{fea}" for fea in ge.columns[1:]})
        return ge

    ge = update_gene_expression(ge, fea_sep)
    # ------------------------------------------------------
    # Create feature scaler
    # ------------------------------------------------------
    # Load and combine responses
    print("Create feature scaler.")
    benchmark.set_split_type(drp.SplitType.TRAIN)
    rsp_tr = benchmark.get_dataframe(drp.SingleDRPDataFrame.RESPONSE)

    benchmark.set_split_type(drp.SplitType.VALIDATION)
    rsp_vl = benchmark.get_dataframe(drp.SingleDRPDataFrame.RESPONSE)

    rsp = pd.concat([rsp_tr, rsp_vl], axis=0)

    # Retian feature rows that are present in the y data (response dataframe)
    # Intersection of omics features, drug features, and responses
    rsp = rsp.merge(ge[benchmark.CANCER_COL_NAME], on=benchmark.CANCER_COL_NAME, how="inner")
    rsp = rsp.merge(md[benchmark.DRUG_COL_NAME], on=benchmark.DRUG_COL_NAME, how="inner")
    ge_sub = ge[ge[benchmark.CANCER_COL_NAME].isin(rsp[benchmark.CANCER_COL_NAME])].reset_index(drop=True)
    md_sub = md[md[benchmark.DRUG_COL_NAME].isin(rsp[benchmark.DRUG_COL_NAME])].reset_index(drop=True)

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

    for stage in drp.SplitType:

        # --------------------------------
        # [Req] Load response data
        # --------------------------------
        benchmark.set_split_type(stage)
        rsp = benchmark.get_dataframe(drp.SingleDRPDataFrame.RESPONSE)
        ge_sub = benchmark.get_dataframe(drp.SingleDRPDataFrame.CELL_LINE_GENE_EXPRESSION).reset_index(drop=True)
        md_sub = benchmark.get_dataframe(drp.SingleDRPDataFrame.DRUG_MORDRED).reset_index(drop=False)

        ge_sub = update_gene_expression(ge_sub, fea_sep)

        # --------------------------------
        # Data prep
        # --------------------------------
        # Retain (canc, drug) responses for which both omics and drug features
        # are available.
        rsp = rsp.merge(ge_sub[benchmark.CANCER_COL_NAME], on=benchmark.CANCER_COL_NAME, how="inner")
        rsp = rsp.merge(md_sub[benchmark.DRUG_COL_NAME], on=benchmark.DRUG_COL_NAME, how="inner")


        # Scale features
        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler) # scale gene expression
        md_sc, _ = scale_df(md_sub, scaler=md_scaler) # scale Mordred descriptors
     
        # --------------------------------
        # [Req] Save ML data files in params["output_dir"]
        # The implementation of this step depends on the model.
        # --------------------------------
        # [Req] Build data name
        
        print("Merge data")
        data = rsp.merge(ge_sc, on=benchmark.CANCER_COL_NAME, how="inner")
        data = data.merge(md_sc, on=benchmark.DRUG_COL_NAME, how="inner")
        data = data.sample(frac=1.0).reset_index(drop=True) # shuffle

        print("Save data")
        file_type = params["feature_data_format"]
        data_fname = f'{benchmark.get_state_string()}.{file_type}'
        if "study" in data.columns:
            data = data.drop(columns=["study"]) # to_parquet() throws error since "study" contain mixed values
        data.to_parquet(Path(params["output_dir"])/data_fname) # saves ML data file to parquet

        # Prepare the y dataframe for the current stage
        fea_list = ["ge", "mordred"]
        fea_cols = [c for c in data.columns if (c.split(fea_sep)[0]) in fea_list]
        meta_cols = [c for c in data.columns if (c.split(fea_sep)[0]) not in fea_list]
        ydf = data[meta_cols]

        # [Req] Save y dataframe for the current stage
        params["y_data_suffix"] = str(benchmark.get_metric())
        frm.save_stage_ydf(ydf, params, stage)

    return params["output_dir"]


# [Req]
def main(args):
    """ Main function to run data preprocessing."""

    # Additional definitions
    additional_definitions = preprocess_params
    # Initialize Config and CLI
    pp = Preprocess()

    params = pp.initialize_parameters(
        filepath,
        default_model="lgbm_params.txt",
        # default_model="params_ws.txt",
        # default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    ml_data_outdir = run(pp,params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
