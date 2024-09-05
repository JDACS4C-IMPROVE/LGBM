""" Sequential implementation of cross-study analysis workflow """

import os
import subprocess
import warnings
from time import time
from pathlib import Path

import pandas as pd

# IMPROVE imports
# from improvelib.initializer.config import Config
# from improvelib.initializer.stage_config import PreprocessConfig, TrainConfig, InferConfig
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
# from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
# from improvelib.applications.drug_response_prediction.config import DRPInferConfig
import improvelib.utils as frm


def build_split_fname(source: str, split: int, phase: str):
    """ Build split file name. If file does not exist continue """
    return f"{source_data_name}_split_{split}_{phase}.txt"


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

print_fn = print
print_fn(f"File path: {filepath}")

# ===============================================================
###  CSA settings
# ===============================================================
# TODO make it work!
# cfg = Config()
# params = cfg.initialize_parameters(
#     pathToModelDir=filepath,
#     section='DEFAULT',
#     default_config="csa_params.txt",
#     default_model=None,
#     additional_definitions=None,
#     required=None
# )
# params = frm.build_paths(params)

cfg = DRPPreprocessConfig() # TODO submit github issue; too many logs printed; is it necessary?
params = cfg.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_params.txt",
    default_model=None,
    additional_cli_section=None,
    additional_definitions=None,
    required=None
)
params = frm.build_paths(params) # TODO move this to improvelib

# Model scripts
model_name = 'lgbm'
preprocess_python_script = f'{model_name}_preprocess_improve.py'
train_python_script = f'{model_name}_train_improve.py'
infer_python_script = f'{model_name}_infer_improve.py'

# Specify dirs
# y_col_name = "auc"
y_col_name = params['y_col_name']
MAIN_CSA_OUTDIR = Path(f"./z_csa_run") # main output dir
MAIN_ML_DATA_DIR = MAIN_CSA_OUTDIR / 'ml_data' # output_dir_pp, input_dir_train, input_dir_infer
MAIN_MODEL_DIR = MAIN_CSA_OUTDIR / 'models' # output_dir_train, input_dir_infer
MAIN_INFER_DIR = MAIN_CSA_OUTDIR / 'infer' # output_dir infer

splits_dir = Path(params['input_dir']) / params['splits_dir']

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


# ===============================================================
###  Generate CSA results (within- and cross-study)
# ===============================================================

timer = Timer()
# Iterate over source datasets
# Note! The "source_data_name" iterations are independent of each other
print_fn(f"\nsource_datasets: {source_datasets}")
print_fn(f"target_datasets: {target_datasets}")
print_fn(f"split_nums:      {split_nums}")

for source_data_name in source_datasets:

    # Get the split file paths
    # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
    if len(split_nums) == 0:
        # Get all splits
        split_files = list((splits_dir).glob(f"{source_data_name}_split_*.txt"))
        split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        split_nums = sorted(set(split_nums))
    else:
        # Use the specified splits
        split_files = []
        for s in split_nums:
            split_files.extend(list((splits_dir).glob(f"{source_data_name}_split_{s}_*.txt")))

    files_joined = [str(s) for s in split_files]

    # --------------------
    # Preprocess and Train
    # --------------------
    for split in split_nums:
        print_fn(f"Split id {split} out of {len(split_nums)} splits.")
        # Check that train, val, and test are available. Else, continue to the next split.
        for phase in ["train", "val", "test"]:
            fname = build_split_fname(source_data_name, split, phase)
            if fname not in "\t".join(files_joined):
                warnings.warn(f"\nThe {phase} split file {fname} is missing \
                              (continue to next split)")
                continue

        for target_data_name in target_datasets:
            if only_cross_study and (source_data_name == target_data_name):
                continue # only cross-study
            print_fn(f"\nSource data: {source_data_name}")
            print_fn(f"Target data: {target_data_name}")

            ml_data_dir = MAIN_ML_DATA_DIR / f"{source_data_name}-{target_data_name}" / \
                f"split_{split}"
            model_dir = MAIN_MODEL_DIR / f"{source_data_name}" / f"split_{split}"
            infer_dir = MAIN_INFER_DIR / f"{source_data_name}-{target_data_name}" / \
                f"split_{split}"

            if source_data_name == target_data_name:
                # If source and target are the same, then infer on the test split
                test_split_file = f"{source_data_name}_split_{split}_test.txt"
            else:
                # If source and target are different, then infer on the entire target dataset
                test_split_file = f"{target_data_name}_all.txt"

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # p1 (none): Preprocess train data
            # train_split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_0_train*.txt"))  # placeholder for LC
            timer_preprocess = Timer()
            # ml_data_path = graphdrp_preprocess_improve.main([
            #     "--train_split_file", f"{source_data_name}_split_{split}_train.txt",
            #     "--val_split_file", f"{source_data_name}_split_{split}_val.txt",
            #     "--test_split_file", str(test_split_file_name),
            #     "--input_dir", str(input_dir),
            #     "--output_dir", str(output_dir),
            #     "--y_col_name", y_col_name
            # ])
            print_fn("\nPreprocessing")
            train_split_file = f"{source_data_name}_split_{split}_train.txt"
            val_split_file = f"{source_data_name}_split_{split}_val.txt"
            print_fn(f"train_split_file: {train_split_file}")
            print_fn(f"val_split_file:   {val_split_file}")
            print_fn(f"test_split_file:  {test_split_file}")
            preprocess_run = ["python", preprocess_python_script,
                  "--train_split_file", str(train_split_file),
                  "--val_split_file", str(val_split_file),
                  "--test_split_file", str(test_split_file),
                  "--input_dir", str("./csa_data/raw_data"),
                  "--output_dir", str(ml_data_dir),
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
            if model_dir.exists() is False:
                timer_train = Timer()
                print_fn("\nTrain")
                print_fn(f"ml_data_dir: {ml_data_dir}")
                print_fn(f"model_dir:   {model_dir}")
                train_run = ["python", train_python_script,
                      "--input_dir", str(ml_data_dir),
                      "--output_dir", str(model_dir),
                      "--y_col_name", y_col_name
                ]
                result = subprocess.run(train_run, capture_output=True,
                                        text=True, check=True)
                timer_train.display_timer(print_fn)

            # Infer
            # p3 (p1, p2): Inference
            timer_infer = Timer()
            infer_run = ["python", infer_python_script,
                  "--input_data_dir", str(ml_data_dir),
                  "--input_model_dir", str(model_dir),
                  "--output_dir", str(infer_dir),
                  "--y_col_name", y_col_name,
                  "--calc_infer_scores", "true"
            ]
            result = subprocess.run(infer_run, capture_output=True,
                                    text=True, check=True)
            timer_infer.display_timer(print_fn)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

timer.display_timer(print_fn)
print_fn('Finished full cross-study run.')
