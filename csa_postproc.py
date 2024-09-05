"""
python csa_postproc.py --res_dir auc_old --model_name GraphDRP --y_col_name auc
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# IMPROVE/CANDLE imports
# from improve import framework as frm
# from improve.csa import cross_study_postprocess
from improvelib.workflow_utils.cross_study.csa_utils import csa_postprocess, plot_color_coded_csa_table


# Imports from preprocess script
# from graphdrp_preprocess_improve import preprocess_params

filepath = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--res_dir', required=False, default='auc', type=str,
                    help='Dir containing the results.')
parser.add_argument('--model_name', required=False, default='LGBM', type=str,
                    help='Name of the model.')
parser.add_argument('--y_col_name', required=False, default='auc', type=str,
                    help='Y col name.')
args = parser.parse_args()

res_dir = args.res_dir
model_name = args.model_name
y_col_name = args.y_col_name

res_dir_path = filepath / res_dir
outdir = res_dir_path / f'../res.csa.{model_name}.{res_dir}'

scores = csa_postprocess(res_dir_path,
                         model_name,
                         y_col_name,
                         outdir=outdir)

# breakpoint()
# scores_filename = Path('r2_mean_csa_table.csv')
# df = pd.read_csv(outdir / scores_filename)
# image_filepath = outdir / scores_filename.with_suffix('.png') # "csa_plot_new.png"
# plot_color_coded_csa_table(df, filepath=image_filepath,
#                            title=scores_filename.with_suffix(''))

# breakpoint()
print('\nFinished cross-study post-processing.')
