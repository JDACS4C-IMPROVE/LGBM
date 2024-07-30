""" Python implementation of cross-study analysis workflow """

import os
import subprocess
import warnings
from time import time
from pathlib import Path
from pprint import pprint

import pandas as pd

# IMPROVE imports
from improvelib.initializer.stage_config import PreprocessConfig, TrainConfig, InferConfig
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
import improvelib.utils as frm

# Model-specific imports (LightGBM)
import lgbm_preprocess_improve
import lgbm_train_improve
import lgbm_infer_improve

class Timer:
  """ Measure time. """
  def __init__(self):
    self.start = time()

  def timer_end(self):
    self.end = time()
    return self.end - self.start

  def display_timer(self, print_fn=print):
    time_diff = self.timer_end()
    if (time_diff) // 3600 > 0:
        print_fn("Runtime: {:.1f} hrs".format( (time_diff)/3600) )
    else:
        print_fn("Runtime: {:.1f} mins".format( (time_diff)/60) )


filepath = Path(__file__).resolve().parent

# breakpoint()
# cfg_pp = PreprocessConfig()
cfg_pp = DRPPreprocessConfig() # TODO submit github issue; too many logs printed; is it necessary?
params_pp = cfg_pp.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_params.txt",
    default_model=None,
    additional_cli_section=None,
    additional_definitions=None,
    required=None
)
params_pp = frm.build_paths(params_pp) # TODO this may move to improvelib

# breakpoint()
# cfg_train = TrainConfig()
cfg_train = DRPTrainConfig()
params_train = cfg_train.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_params.txt",
    default_model=None,
    additional_cli_section=None,
    additional_definitions=None,
    required=None
)

# breakpoint()
# cfg_infer = InferConfig()
cfg_infer = DRPInferConfig()
params_infer = cfg_infer.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_params.txt",
    default_model=None,
    additional_cli_section=None,
    additional_definitions=None,
    required=None
)

# breakpoint()
print("Params preprocess:"); pprint(params_pp)
print("Params train:"); pprint(params_train)
print("Params infer:"); pprint(params_infer)

# input_dir_pp = 
# output_dir_pp = 
# input_dir_train = 
# output_dir_train = 
# input_dir_infer = 
# output_dir_infer = 

# y_col_name = "auc"
# y_col_name = "auc1"
y_col_name = params_pp["y_col_name"] #"auc"

# maindir = Path(f"./{y_col_name}")
maindir = Path(f"./0_{y_col_name}_improvelib") # main output dir
# Note! ML data and trained model should be saved to the same dir for inference script
MAIN_ML_DATA_DIR = maindir / "ml_data" # output_dir_pp, input_dir_train, input_dir_infer
MAIN_MODEL_DIR = maindir / "models" # output_dir_train, input_dir_infer
# MAIN_MLDATA_AND_MODEL_DIR = Path(f"./{maindir}/mldata_and_model")
# MAIN_INFER_OUTDIR = Path(f"./{maindir}/infer") # output_dir infer
MAIN_INFER_DIR = maindir / "infer" # output_dir infer

# main_datadir = Path(os.environ["IMPROVE_DATA_DIR"])
# raw_datadir = main_datadir / params["raw_data_dir"]
# x_datadir = raw_datadir / params["x_data_dir"]
# y_datadir = raw_datadir / params["y_data_dir"]
# splits_dir = raw_datadir / params["splits_dir"]
splits_dir = Path(params_pp["input_dir"]) / params_pp["splits_dir"]

# lg = Logger(main_datadir/"csa.log")
print_fn = print
# print_fn = get_print_func(lg.logger)
print_fn(f"File path: {filepath}")

### Source and target data sources
## Set 1 - full analysis
source_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
target_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
## Set 2 - smaller datasets
# source_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
# source_datasets = ["CCLE", "GDSCv1"]
# target_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
## Set 3 - full analysis for a single source
# source_datasets = ["CCLE"]
# source_datasets = ["CTRPv2"]
# target_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "gCSI", "GDSCv2"]
## Set 4 - same source and target
# source_datasets = ["CCLE"]
# target_datasets = ["CCLE"]
## Set 5 - single source and target
# source_datasets = ["GDSCv1"]
# target_datasets = ["CCLE"]

only_cross_study = False
# only_cross_study = True

## Splits
split_nums = []  # all splits
# split_nums = [0]
# split_nums = [4, 7]
# split_nums = [1, 4, 7]
# split_nums = [1, 3, 5, 7, 9]

def build_split_fname(source: str, split: int, phase: str):
    """ Build split file name. If file does not exist continue """
    return f"{source_data_name}_split_{split}_{phase}.txt"

# ===============================================================
###  Generate CSA results (within- and cross-study)
# ===============================================================

timer = Timer()
# Iterate over source datasets
# Note! The "source_data_name" iterations are independent of each other
print_fn(f"\nsource_datasets: {source_datasets}")
print_fn(f"target_datasets: {target_datasets}")
print_fn(f"split_nums:      {split_nums}")
# breakpoint()
for source_data_name in source_datasets:

    # Get the split file paths
    # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
    if len(split_nums) == 0:
        # Get all splits
        split_files = list((splits_dir).glob(f"{source_data_name}_split_*.txt"))
        split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        split_nums = sorted(set(split_nums))
        # num_splits = 1
    else:
        # Use the specified splits
        split_files = []
        for s in split_nums:
            split_files.extend(list((splits_dir).glob(f"{source_data_name}_split_{s}_*.txt")))

    files_joined = [str(s) for s in split_files]

    # --------------------
    # Preprocess and Train
    # --------------------
    # breakpoint()
    for split in split_nums:
        print_fn(f"Split id {split} out of {len(split_nums)} splits.")
        # Check that train, val, and test are available. Otherwise, continue to the next split.
        # split = 11
        # files_joined = [str(s) for s in split_files]
        # TODO: check this!
        for phase in ["train", "val", "test"]:
            fname = build_split_fname(source_data_name, split, phase)
            # print(f"{phase}: {fname}")
            if fname not in "\t".join(files_joined):
                warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
                continue

        for target_data_name in target_datasets:
            if only_cross_study and (source_data_name == target_data_name):
                continue # only cross-study
            print_fn(f"\nSource data: {source_data_name}")
            print_fn(f"Target data: {target_data_name}")

            # ml_data_outdir = MAIN_ML_DATA_DIR/f"{source_data_name}-{target_data_name}"/f"split_{split}" # AP
            # mldata_and_model_dir = MAIN_MLDATA_AND_MODEL_DIR/f"{source_data_name}-{target_data_name}"/f"split_{split}" # AP
            ml_data_dir = MAIN_ML_DATA_DIR / f"{source_data_name}-{target_data_name}"/f"split_{split}" # AP
            model_dir = MAIN_MODEL_DIR / f"{source_data_name}"/f"split_{split}" # AP
            infer_dir = MAIN_INFER_DIR / f"{source_data_name}-{target_data_name}" / f"split_{split}" # AP

            if source_data_name == target_data_name:
                # If source and target are the same, then infer on the test split
                test_split_file = f"{source_data_name}_split_{split}_test.txt"
            else:
                # If source and target are different, then infer on the entire target dataset
                test_split_file = f"{target_data_name}_all.txt"

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # p1 (none): Preprocess train data
            # train_split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_0_train*.txt"))  # TODO placeholder for lc analysis
            timer_preprocess = Timer()
            # ml_data_path = graphdrp_preprocess_improve.main([
            #     "--train_split_file", f"{source_data_name}_split_{split}_train.txt",
            #     "--val_split_file", f"{source_data_name}_split_{split}_val.txt",
            #     "--test_split_file", str(test_split_file_name),
            #     "--ml_data_outdir", str(ml_data_outdir),
            #     "--y_col_name", y_col_name
            # ])
            print_fn("\nPreprocessing")
            train_split_file = f"{source_data_name}_split_{split}_train.txt"
            val_split_file = f"{source_data_name}_split_{split}_val.txt"
            print_fn(f"train_split_file: {train_split_file}")
            print_fn(f"val_split_file:   {val_split_file}")
            print_fn(f"test_split_file:  {test_split_file}")
            # print_fn(f"ml_data_outdir:   {ml_data_outdir}") # AP
            print_fn(f"ml_data_dir:      {ml_data_dir}") # AP
            preprocess_run = ["python",
                  "lgbm_preprocess_improve.py",
                  "--train_split_file", str(train_split_file),
                  "--val_split_file", str(val_split_file),
                  "--test_split_file", str(test_split_file),
                  # "--ml_data_outdir", str(ml_data_outdir), # AP
                  "--input_dir", str("./csa_data/raw_data"), # AP
                  "--output_dir", str(ml_data_dir), # AP
                  "--y_col_name", str(y_col_name)
            ]
            result = subprocess.run(preprocess_run, capture_output=True,
                                    text=True, check=True)
            # print(result.stdout)
            # print(result.stderr)
            timer_preprocess.display_timer(print_fn)

            # p2 (p1): Train model
            # Train a single model for a given [source, split] pair
            # Train using train samples and early stop using val samples
            # model_outdir = MAIN_MODEL_DIR/f"{source_data_name}"/f"split_{split}" # AP
            # if model_outdir.exists() is False: # AP
            if model_dir.exists() is False: # AP
                # train_ml_data_dir = ml_data_outdir # AP
                # val_ml_data_dir = ml_data_outdir # AP
                timer_train = Timer()
                # graphdrp_train_improve.main([
                #     "--train_ml_data_dir", str(train_ml_data_dir),
                #     "--val_ml_data_dir", str(val_ml_data_dir),
                #     "--model_outdir", str(model_outdir),
                #     "--epochs", str(epochs),  # available in config_file
                #     # "--ckpt_directory", str(MODEL_OUTDIR),  # TODO: we'll use candle known param ckpt_directory instead of model_outdir
                #     # "--cuda_name", "cuda:5"
                # ])
                print_fn("\nTrain")
                # print_fn(f"train_ml_data_dir: {train_ml_data_dir}") # AP
                # print_fn(f"val_ml_data_dir:   {val_ml_data_dir}") # AP
                # print_fn(f"model_outdir:      {model_outdir}") # AP
                print_fn(f"ml_data_dir: {ml_data_dir}") # AP
                print_fn(f"model_dir:   {model_dir}") # AP
                # breakpoint()
                train_run = ["python",
                      "lgbm_train_improve.py",
                      # "--train_ml_data_dir", str(train_ml_data_dir), # AP
                      # "--val_ml_data_dir", str(val_ml_data_dir), # AP
                      # "--model_outdir", str(model_outdir), # AP
                      "--input_dir", str(ml_data_dir),
                      "--output_dir", str(model_dir),
                      "--y_col_name", y_col_name
                ]
                result = subprocess.run(train_run, capture_output=True,
                                        text=True, check=True)
                # print(result.stdout)
                # print(result.stderr)
                timer_train.display_timer(print_fn)

            # Infer
            # p3 (p1, p2): Inference
            # test_ml_data_dir = ml_data_outdir # AP
            # model_dir = model_outdir # AP
            # infer_outdir = MAIN_INFER_OUTDIR/f"{source_data_name}-{target_data_name}"/f"split_{split}" # AP
            timer_infer = Timer()
            # graphdrp_infer_improve.main([
            #     "--test_ml_data_dir", str(test_ml_data_dir),
            #     "--model_dir", str(model_dir),
            #     "--infer_outdir", str(infer_outdir),
            #     # "--cuda_name", "cuda:5"
            # ])
            print_fn("\nInfer")
            # print_fn(f"test_ml_data_dir: {test_ml_data_dir}") # AP
            # print_fn(f"infer_outdir:     {infer_outdir}") # AP
            infer_run = ["python",
                  "lgbm_infer_improve.py",
                  # "--test_ml_data_dir", str(test_ml_data_dir), # AP
                  # "--model_dir", str(model_dir), # AP
                  # "--infer_outdir", str(infer_outdir), # AP
                  "--input_dir_data", str(ml_data_dir), # AP
                  "--input_dir_model", str(model_dir), # AP
                  "--output_dir", str(infer_dir), # AP
                  "--y_col_name", y_col_name
            ]
            result = subprocess.run(infer_run, capture_output=True,
                                    text=True, check=True)
            timer_infer.display_timer(print_fn)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

timer.display_timer(print_fn)
print_fn("Finished a full cross-study run.")
