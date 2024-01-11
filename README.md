This repo demonstrates the use of [improve](https://github.com/JDACS4C-IMPROVE/IMPROVE) for drug response prediction (DRP) with LightGBM and the benchmark [cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

# Dependencies
Check `conda_env.sh`
+ [candle-lib](https://github.com/ECP-CANDLE/candle_lib) -- improve lib dependency
+ [LightGBM](https://lightgbm.readthedocs.io/en/stable/) -- ML model
+ [Pyarrow](https://anaconda.org/conda-forge/pyarrow) -- allows to save/load parquet files

## Source codes
+ `lgbm_preprocess_improve.py`: creates data files for drug resposne prediction (DRP)
+ `lgbm_train_improve.py`: trains a LightGBM DRP model
+ `lgbm_infer_improve.py`: runs inference with the trained LightGBM model
+ `lgbm_params.txt`: parameter file

# Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).
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

# Step-by-step running

### 1. Clone the repo
```
git clone https://github.com/JDACS4C-IMPROVE/LGBM/tree/develop
cd LGBM
```

### 2. Download benchmark data
```
sh ./download_csa.sh
```

### 3. Set computational environment
* Install dependencies (check `conda_env.sh`)
* Set the required environment variables to point towards the data folder and improve lib. You need to download the improve lib repo (follow this repo for more info `https://github.com/JDACS4C-IMPROVE/IMPROVE`).
```bash
export IMPROVE_DATA_DIR="./csa_data/"
export PYTHONPATH=$PYTHONPATH:/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE
```

### 4. Preprocess benchmark data (_raw data_) to construct model input data (_ML data_)
```bash
python lgbm_preprocess_improve.py
```
Generates:
* three model input data files: `train_data.parquet`, `val_data.parquet`, `test_data.parquet`
* three data files, each containing y data (responses) and metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

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
Trains LightGBM using the processed data: `train_data.parquet` (training), `val_data.parquet` (early stopping).

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

### 6. Run the trained model in inference mode on test data
```python lgbm_infer_improve.py```
This script uses processed data and the trained model to evaluate performance.

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
