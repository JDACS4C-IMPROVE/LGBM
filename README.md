# LGBM

This repository demonstrates how to use the [IMPROVE library v0.1.0](https://jdacs4c-improve.github.io/docs/v0.1.0-alpha/) for building a drug response prediction (DRP) model using LightGBM (LGBM), and provides examples with the benchmark [cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

This version, tagged as `v0.1.0-2024-09-27`, introduces a new API which is designed to encourage broader adoption of IMPROVE and its curated models by the research community.



## Dependencies
Installation instuctions are detialed below in [Step-by-step instructions](#step-by-step-instructions).

Conda `yml` file [conda_wo_candle.yml](./conda_wo_candle.yml)

ML framework:
+ [LightGBM](https://lightgbm.readthedocs.io/en/stable/) - machine learning framework for building the prediction model
+ [pyarrow](https://anaconda.org/conda-forge/pyarrow) - save and load parquet files

IMPROVE dependencies:
+ [IMPROVE tag v0.1.0-2024-09-27](https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.1.0-2024-09-27)



## Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

The data tree is shown below:
```
csa_data/raw_data/
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   └── GDSCv2_split_9_val.txt
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   └── drug_SMILES.tsv
└── y_data
    └── response.tsv
```



## Model scripts and parameter file
+ `lgbm_preprocess_improve.py` - takes benchmark data files and transforms into files for trianing and inference
+ `lgbm_train_improve.py` - trains a LightGBM-based DRP model
+ `lgbm_infer_improve.py` - runs inference with the trained LightGBM model
+ `model_params_def.py` - definitions of parameters that are specific to the model
+ `lgbm_params.txt` - default parameter file (parameter values specified in this file override the defaults)



# Step-by-step instructions

### 1. Clone the model repository and checkout the tag
```
git clone git@github.com:JDACS4C-IMPROVE/LGBM.git
cd LGBM
git checkout v0.1.0-2024-09-27
```

### 2. Set computational environment
Option 1: create conda env using `yml`
```bash
conda env create -f conda_wo_candle.yml
```

Option 2: use [conda_env_py37.sh](./conda_env_py37.sh)

Option 3: use these commands
```bash
CONDA_ENV_NAME=lgbm_py37
conda create -n $CONDA_ENV_NAME python=3.7 pip lightgbm=3.1.1 --yes
conda activate $CONDA_ENV_NAME
conda install conda-forge::pyarrow
```

### 3. Run `setup_improve.sh`.
```bash
source setup_improve.sh
```

This will:
1. Download cross-study analysis (CSA) benchmark data into `./csa_data/`.
2. Clone IMPROVE repo (checkout `v0.1.0-2024-09-27`) outside the LGBM model repo.
3. Set up `PYTHONPATH` (adds IMPROVE repo).


### 4. Preprocess CSA benchmark data (_raw data_) to construct model input data (_ML data_)
```bash
python lgbm_preprocess_improve.py --input_dir ./csa_data/raw_data --output_dir exp_result
```

Preprocesses the CSA data and creates train, validation (val), and test datasets.

Generates:
* three model input data files: `train_data.parquet`, `val_data.parquet`, `test_data.parquet`
* three tabular data files, each containing the drug response values (i.e. AUC) and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

```
exp_result
├── param_log_file.txt
├── test_data.parquet
├── test_y_data.csv
├── train_data.parquet
├── train_y_data.csv
├── val_data.parquet
├── val_y_data.csv
├── x_data_gene_expression_scaler.gz
└── x_data_mordred_scaler.gz
```


### 5. Train LightGBM model
```bash
python lgbm_train_improve.py --input_dir exp_result --output_dir exp_result
```

Trains a LightGBM model using the model input data: `train_data.parquet` (training), `val_data.parquet` (early stopping).

Generates:
* trained model: `model.txt`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`
```
exp_result
├── model.txt
├── param_log_file.txt
├── test_data.parquet
├── test_y_data.csv
├── train_data.parquet
├── train_y_data.csv
├── val_data.parquet
├── val_scores.json
├── val_y_data.csv
├── val_y_data_predicted.csv
├── x_data_gene_expression_scaler.gz
└── x_data_mordred_scaler.gz
```


### 6. Run inference on test data with the trained LightGBM model
```bash
python lgbm_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
exp_result
├── model.txt
├── param_log_file.txt
├── test_data.parquet
├── test_scores.json
├── test_y_data.csv
├── test_y_data_predicted.csv
├── train_data.parquet
├── train_y_data.csv
├── val_data.parquet
├── val_scores.json
├── val_y_data.csv
├── val_y_data_predicted.csv
├── x_data_gene_expression_scaler.gz
└── x_data_mordred_scaler.gz
```
