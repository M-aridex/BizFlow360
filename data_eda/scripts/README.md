# Scripts

This directory contains reusable Python scripts used during the
data preparation stage of the SME Financial Distress Prediction
project.

## Files

### clean_msme_data.py

Loads the KNBS MSME Stata dataset, performs basic structural
cleaning and saves the cleaned MSME dataset.

### clean_world_bank_data.py

Loads the World Bank Enterprise Survey data and performs basic
structural cleaning.

### engineer_financial_features.py

Creates the financial variables used during financial analysis
and machine learning preparation.

Key derived variables include:

- `eh01_1_clean`
- `eh15_1_clean`
- `eh04_1_clean`
- `net_income_margin`
- `revenue_change_ratio`
- `business_closed`
- `business_closure_count`

### build_model_dataset.py

Creates a modeling-oriented dataset from the feature-engineered
MSME data.

## Execution order

The general pipeline is:

1. `clean_msme_data.py`
2. `clean_world_bank_data.py`
3. `engineer_financial_features.py`
4. `build_model_dataset.py`

## Important

These scripts are preparation utilities.

The actual XGBoost model training should be handled separately by
the machine-learning/modeling stage.