# LGBM

This repository demonstrates the use of the [`IMPROVE library`](https://github.com/JDACS4C-IMPROVE/IMPROVE) for drug response prediction (DRP) with LightGBM (LGBM) and the benchmark [cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

A more detailed tutorial can be found [here](https://jdacs4c-improve.github.io/docs/content/unified_interface.html). 

## Dependencies
+ [candle-lib](https://github.com/ECP-CANDLE/candle_lib) - enables various hyperparameter optimization (HPO) schemes and automatically distributes the workload across available computing resources
+ [LightGBM](https://lightgbm.readthedocs.io/en/stable/) - machine learning model
+ [pyarrow](https://anaconda.org/conda-forge/pyarrow) - saves and loads parquet files
+ [IMPROVE](https://github.com/JDACS4C-IMPROVE/IMPROVE) - contains scripts specific for the IMPROVE project 

## Source codes
+ `lgbm_preprocess_improve.py`: creates data files for the DRP model
+ `lgbm_train_improve.py`: trains a DRP model using LightGBM
+ `lgbm_infer_improve.py`: runs inference with the trained LightGBM model
+ `lgbm_params.txt`: parameter file

## Dataset
Benchmark data for CSA can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

The required data tree is shown below:

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

## Step-by-step running

### 1. Clone the repository
```
git clone https://github.com/JDACS4C-IMPROVE/LGBM
cd LGBM
git checkout develop
```

### 2. Download CSA data
```
sh ./download_csa.sh
```

### 3. Set up the environment

Install dependencies:
```bash
conda create -n lgbm_py37 python=3.7 pip lightgbm=3.1.1 --yes
conda activate lgbm_py37
pip install pyarrow==12.0.1
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

Clone the `IMPROVE library` (outside of the LGBM folder):
```bash
cd ..
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
export MY_PATH_TO_IMPROVE=`pwd`
cd ..
```

Set the required environment variables to point towards the location of the data folder and `IMPROVE library`:
```bash
cd LGBM
export IMPROVE_DATA_DIR="./csa_data/"
export PYTHONPATH=$PYTHONPATH:${MY_PATH_TO_IMPROVE}
```

### 4. Preprocess CSA data (_raw data_) to construct model input data (_ML data_)
```bash
python lgbm_preprocess_improve.py
```

Preprocesses the CSA data into train, validation (val), and test datasets. 

Generates:
* three model input data files: `train_data.parquet`, `val_data.parquet`, `test_data.parquet`
* three y data files, each containing the drug response values (i.e. AUC) and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

```
ml_data
└── CCLE-CCLE
    └── split_0
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
python lgbm_train_improve.py
```
Trains a LightGBM model using the ML data: `train_data.parquet` (training), `val_data.parquet` (early stopping).

Generates:
* trained model: `model.txt`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`
```
out_models
└── CCLE
    └── split_0
        ├── model.txt
        ├── val_scores.json
        └── val_y_data_predicted.csv
```

### 6. Run inference on test data with trained LightGBM model
```python lgbm_infer_improve.py```

Evaluates the performance of a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
out_infer
└── CCLE-CCLE
    └── split_0
        ├── test_scores.json
        └── test_y_data_predicted.csv
```
