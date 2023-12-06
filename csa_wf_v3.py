""" Python implementation of cross-study analysis workflow """
# cuda_name = "cuda:6"
# cuda_name = "cuda:7"

import os
import warnings
import subprocess
from pathlib import Path
import pandas as pd

# IMPROVE imports
from improve import framework as frm
# import improve_utils
# from improve_utils import improve_globals as ig

# LightGBM imports
import lgbm_preprocess
import lgbm_train
import lgbm_infer

# from ap_utils.classlogger import Logger
from ap_utils.utils import get_print_func, Timer

fdir = Path(__file__).resolve().parent

# ML_DATA_DIR = Path("./ml_data")
y_col_name = "auc"
# y_col_name = "auc1"
maindir = Path(f"./{y_col_name}")
MAIN_ML_DATA_DIR = Path(f"./{maindir}/ml_data")
MAIN_MODEL_DIR = Path(f"./{maindir}/models")
MAIN_INFER_OUTDIR = Path(f"./{maindir}/infer")

# Check that environment variable "IMPROVE_DATA_DIR" has been specified
if os.getenv("IMPROVE_DATA_DIR") is None:
    raise Exception("ERROR ! Required system variable not specified.  \
                    You must define IMPROVE_DATA_DIR ... Exiting.\n")
os.environ["CANDLE_DATA_DIR"] = os.environ["IMPROVE_DATA_DIR"]

params = frm.initialize_parameters(
    fdir,
    default_model="csa_workflow_params.txt",
)

main_datadir = Path(os.environ["IMPROVE_DATA_DIR"])
raw_datadir = main_datadir / params["raw_data_dir"]
x_datadir = raw_datadir / params["x_data_dir"]
y_datadir = raw_datadir / params["y_data_dir"]
splits_dir = raw_datadir / params["splits_dir"]

# AP
# lg = Logger(main_datadir/"csa.log")
print_fn = print
# print_fn = get_print_func(lg.logger)
print_fn(f"File path: {fdir}")

### Source and target data sources
## Set 1 - full analysis
# source_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
## Set 2 - smaller datasets
# source_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
# source_datasets = ["CCLE", "GDSCv1"]
# target_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
## Set 3 - full analysis for a single source
# source_datasets = ["CCLE"]
source_datasets = ["CTRPv2"]
target_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "gCSI", "GDSCv2"]
# target_datasets = ["GDSCv1"]
## Set 4 - same source and target
# source_datasets = ["CCLE"]
# target_datasets = ["CCLE"]
## Set 5 - single source and target
# source_datasets = ["CCLE"]
# target_datasets = ["GDSCv1"]

# only_cross_study = False
only_cross_study = True

## Splits
# split_nums = []  # all splits
# split_nums = [0]
# split_nums = [4, 7]
split_nums = [1, 4, 7]
# split_nums = [1, 3, 5, 7, 9]

## Parameters of the experiment/run/workflow
# TODO: this should be stored as the experiment metadata that we can go back check
# config_file_name = "csa_params.txt"
# config_file_path = fdir/config_file_name

def build_split_fname(source, split, phasea):
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
# import pdb; pdb.set_trace()
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
    # import pdb; pdb.set_trace()
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

        # import pdb; pdb.set_trace()
        for target_data_name in target_datasets:
            if only_cross_study and (source_data_name == target_data_name):
                continue
            print_fn(f"\nSource data: {source_data_name}")
            print_fn(f"Target data: {target_data_name}")

            # EXP_ML_DATA_DIR = ig.ml_data_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"
            ml_data_outdir = MAIN_ML_DATA_DIR/f"{source_data_name}-{target_data_name}"/f"split_{split}"

            if source_data_name == target_data_name:
                # If source and target are the same, then infer on the test split
                test_split_file = f"{source_data_name}_split_{split}_test.txt"
            else:
                # If source and target are different, then infer on the entire target dataset
                test_split_file = f"{target_data_name}_all.txt"

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # p1 (none): Preprocess train data
            # import pdb; pdb.set_trace()
            # train_split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_0_train*.txt"))  # TODO: placeholder for lc analysis
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
            # test_split_file = f"{source_data_name}_split_{split}_test.txt"
            print_fn(f"train_split_file: {train_split_file}")
            print_fn(f"val_split_file:   {val_split_file}")
            print_fn(f"test_split_file:  {test_split_file}")
            print_fn(f"ml_data_outdir:   {ml_data_outdir}")
            # import pdb; pdb.set_trace()
            preprocess_run = ["python",
                  # "graphdrp_preprocess_improve.py",
                  "lgbm_preprocess.py",
                  "--train_split_file", str(train_split_file),
                  "--val_split_file", str(val_split_file),
                  "--test_split_file", str(test_split_file),
                  "--ml_data_outdir", str(ml_data_outdir),
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
            # import pdb; pdb.set_trace()
            model_outdir = MAIN_MODEL_DIR/f"{source_data_name}"/f"split_{split}"
            if model_outdir.exists() is False:
                train_ml_data_dir = ml_data_outdir
                val_ml_data_dir = ml_data_outdir
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
                print_fn(f"train_ml_data_dir: {train_ml_data_dir}")
                print_fn(f"val_ml_data_dir:   {val_ml_data_dir}")
                print_fn(f"model_outdir:      {model_outdir}")
                # import pdb; pdb.set_trace()
                train_run = ["python",
                      # "graphdrp_train_improve.py",
                      "lgbm_train.py",
                      "--train_ml_data_dir", str(train_ml_data_dir),
                      "--val_ml_data_dir", str(val_ml_data_dir),
                      "--model_outdir", str(model_outdir),
                      "--y_col_name", y_col_name
                ]
                result = subprocess.run(train_run, capture_output=True,
                                        text=True, check=True)
                # print(result.stdout)
                # print(result.stderr)
                timer_train.display_timer(print_fn)

            # Infer
            # p3 (p1, p2): Inference
            # import pdb; pdb.set_trace()
            test_ml_data_dir = ml_data_outdir
            model_dir = model_outdir
            infer_outdir = MAIN_INFER_OUTDIR/f"{source_data_name}-{target_data_name}"/f"split_{split}"
            timer_infer = Timer()
            # graphdrp_infer_improve.main([
            #     "--test_ml_data_dir", str(test_ml_data_dir),
            #     "--model_dir", str(model_dir),
            #     "--infer_outdir", str(infer_outdir),
            #     # "--cuda_name", "cuda:5"
            # ])
            print_fn("\nInfer")
            print_fn(f"test_ml_data_dir: {test_ml_data_dir}")
            print_fn(f"val_ml_data_dir:  {val_ml_data_dir}")
            print_fn(f"infer_outdir:     {infer_outdir}")
            # import pdb; pdb.set_trace()
            infer_run = ["python",
                  # "graphdrp_infer_improve.py",
                  "lgbm_infer.py",
                  "--test_ml_data_dir", str(test_ml_data_dir),
                  "--model_dir", str(model_dir),
                  "--infer_outdir", str(infer_outdir),
                  "--y_col_name", y_col_name
            ]
            result = subprocess.run(infer_run, capture_output=True,
                                    text=True, check=True)
            timer_infer.display_timer(print_fn)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Note! The "split" iterations are independent of each other
    # for split in split_nums:
    #     print_fn(f"Split {int(split) + 1} (id {split}) out of {len(split_nums)} splits.")
    #     # Check that train, val, and test are available. Otherwise, continue to the next split.
    #     # split = 11
    #     files_joined = [str(s) for s in split_files]
    #     phase = "train"
    #     fname = f"{source_data_name}_split_{split}_{phase}.txt"
    #     if fname not in "\t".join(files_joined):
    #         warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
    #         continue
    #     phase = "val"
    #     fname = f"{source_data_name}_split_{split}_{phase}.txt"
    #     if fname not in "\t".join(files_joined):
    #         warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
    #         continue
    #     phase = "test"
    #     fname = f"{source_data_name}_split_{split}_{phase}.txt"
    #     if fname not in "\t".join(files_joined):
    #         warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
    #         continue

    #     # Iterate over target datasets
    #     for target_data_name in target_datasets:
    #         print_fn(f"\nSource data: {source_data_name}")
    #         print_fn(f"Target data: {target_data_name}\n")

    #         EXP_ML_DATA_DIR = ig.ml_data_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"

    #         if source_data_name == target_data_name:
    #             # If source and target are the same, then infer on the test split
    #             test_split_file_name = f"{source_data_name}_split_{split}_test.txt"
    #         else:
    #             # If source and target are different, then infer on the entire target dataset
    #             test_split_file_name = f"{target_data_name}_all.txt"

    #         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #         # p1 (none): Preprocess train data
    #         # train_split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_0_train*.txt"))  # TODO: placeholder for lc analysis
    #         # import pdb; pdb.set_trace()
    #         timer_preprocess = Timer()
    #         # frm_preprocess.main([
    #         ml_data_path = frm_preprocess_tr_vl_te.main([
    #             "--train_data_name", str(source_data_name),
    #             "--val_data_name", str(source_data_name),
    #             "--test_data_name", str(target_data_name),
    #             "--train_split_file_name", f"{source_data_name}_split_{split}_train.txt",
    #             "--val_split_file_name", f"{source_data_name}_split_{split}_val.txt",
    #             "--test_split_file_name", str(test_split_file_name),
    #             "--y_col_name", y_col_name,
    #             "--outdir", str(EXP_ML_DATA_DIR)
    #         ])
    #         timer_preprocess.display_timer(print_fn)

    #         # p2 (p1): Train model
    #         # Train using train samples and early stop using val samples
    #         TRAIN_ML_DATA_DIR = EXP_ML_DATA_DIR
    #         VAL_ML_DATA_DIR = EXP_ML_DATA_DIR
    #         MODEL_OUTDIR = ig.models_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    #         # import pdb; pdb.set_trace()
    #         timer_train = Timer()
    #         frm_train.main([
    #             # "--config_file", str(config_file_path),  # TODO: we should be able to pass the config_file
    #             "--epochs", str(epochs),  # available in config_file
    #             "--train_ml_data_dir", str(TRAIN_ML_DATA_DIR),
    #             "--val_ml_data_dir", str(VAL_ML_DATA_DIR),
    #             "--model_outdir", str(MODEL_OUTDIR),
    #             # "--ckpt_directory", str(MODEL_OUTDIR),  # TODO: we'll use candle known param ckpt_directory instead of model_outdir
    #             "--model_arch", str(model_arch),  # specific to GraphDRP
    #             "--y_col_name", y_col_name,
    #             "--cuda_name", "cuda:5"
    #         ])
    #         timer_train.display_timer(print_fn)

    #         # p3 (p1, p2): Inference
    #         TEST_ML_DATA_DIR = EXP_ML_DATA_DIR
    #         infer_outdir = ig.infer_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    #         # import pdb; pdb.set_trace()
    #         timer_infer = Timer()
    #         frm_infer.main([
    #             # "--config_file", config_file_path,
    #             "--test_ml_data_dir", str(TEST_ML_DATA_DIR),
    #             "--model_dir", str(MODEL_OUTDIR),
    #             "--infer_outdir", str(infer_outdir),
    #             "--model_arch", str(model_arch),  # specific to GraphDRP
    #             "--y_col_name", y_col_name,
    #             "--cuda_name", "cuda:5"
    #         ])
    #         timer_infer.display_timer(print_fn)


# timer.display_timer(print)
timer.display_timer(print_fn)
print_fn("Finished a full cross-study run.")
