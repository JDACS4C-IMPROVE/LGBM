This example repo demonstrates the use of LightGBM model for drug response prediction with improve lib and cross-study analysis (CSA) dataset.

# Resources:
+ README.md: this file.
+ Data: CSA dataset

## Dependencies
Check `conda_env.sh`
+ [Candle-lib](https://github.com/ECP-CANDLE/candle_lib) -- improve lib dependency
+ [LightGBM](https://lightgbm.readthedocs.io/en/stable/) -- ML model
+ [Pyarrow](https://anaconda.org/conda-forge/pyarrow) -- allows to save/load parquet files

## Source codes
+ lgbm_preprocess_improve.py: create data from cell and drug files.
+ lgbm_train_improve.py: train a LightGBM model.
+ lgbm_infer_improve.py: infer responses with a trained LightGBM model.
+ lgbm_params.txt: parameter file

# Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/)
The required data tree is shown next:

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

# Step-by-step running:

## 1. Define required environment variable to point towards data folder, e.g.
Follow this repo to set up the env variables for IMPROVE_DATA_DIR and improve lib.
https://github.com/JDACS4C-IMPROVE/IMPROVE

## 2. Preprocess raw benchmark data to construct model input data
```python lgbm_preprocess_improve.py```

This generates:
* three model input data sets: train_data.parquet, val_data.parquet, infer_data.parquet
* three data files containing the y data (and metadata): train_y_data.csv, val_y_data.csv, infer_y_data.csv

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

## 3. Train the LightGBM model
```python lgbm_train_improve.py```

This trains LightGBM using the processed data: train_data.parquet (training), val_data.parquet (early stopping).

This generates:
* trained model: model.txt
* prediction on val data: val_y_data_predicted.csv
* prediction performance scores on val data: val_scores.json
```
out_models
└── CCLE
    └── split_0
        ├── model.txt
        ├── val_scores.json
        └── val_y_data_predicted.csv
```

## 4. Run the trained model in inference on test data
```python lgbm_infer_improve.py```

The scripts uses processed data and the trained model to evaluate performance which is stored in files: `test_scores.json` and `test_predicted.csv`.

This generates:
* trained model: model.txt
* prediction on test data: test_y_data_predicted.csv
* prediction performance scores on test data: test_scores.json
```
out_infer
└── CCLE-CCLE
    └── split_0
        ├── test_scores.json
        └── test_y_data_predicted.csv
```
