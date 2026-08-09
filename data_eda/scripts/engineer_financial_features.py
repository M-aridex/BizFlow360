from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "processed" / "msme_analysis_cleaned.csv"

OUTPUT_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "msme_financial_features.csv"


# ---------------------------------------------------------------
# Load cleaned MSME data
# ---------------------------------------------------------------

print("Loading cleaned MSME data...")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)


# ---------------------------------------------------------------
# Financial variables
# ---------------------------------------------------------------

financial_columns = [
    "eh01_1",
    "eh04_1",
    "eh15_1",
    "eh22_1"
]

available = [
    col for col in financial_columns
    if col in df.columns
]

print("Available financial variables:", available)


# ---------------------------------------------------------------
# Convert financial variables to numeric
# ---------------------------------------------------------------

for column in available:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ---------------------------------------------------------------
# Replace invalid negative financial values
# ---------------------------------------------------------------

for column in available:
    df.loc[df[column] < 0, column] = np.nan


# ---------------------------------------------------------------
# Create cleaned financial variables
# ---------------------------------------------------------------

if "eh01_1" in df.columns:
    df["eh01_1_clean"] = df["eh01_1"]

if "eh15_1" in df.columns:
    df["eh15_1_clean"] = df["eh15_1"]

if "eh04_1" in df.columns:
    df["eh04_1_clean"] = df["eh04_1"]


# ---------------------------------------------------------------
# Net income margin
# ---------------------------------------------------------------

if {
    "eh04_1_clean",
    "eh01_1_clean"
}.issubset(df.columns):

    df["net_income_margin"] = (
        df["eh04_1_clean"]
        / df["eh01_1_clean"]
    )


# ---------------------------------------------------------------
# Revenue change ratio
# ---------------------------------------------------------------

if {
    "eh01_1_clean",
    "eh15_1_clean"
}.issubset(df.columns):

    df["revenue_change_ratio"] = (
        df["eh01_1_clean"]
        / df["eh15_1_clean"]
    )


# ---------------------------------------------------------------
# Replace infinite values
# ---------------------------------------------------------------

derived_columns = [
    "net_income_margin",
    "revenue_change_ratio"
]

existing_derived = [
    col for col in derived_columns
    if col in df.columns
]

df[existing_derived] = df[existing_derived].replace(
    [np.inf, -np.inf],
    np.nan
)


# ---------------------------------------------------------------
# Business closure variables
# ---------------------------------------------------------------

if "en01" in df.columns:

    df["business_closed"] = (
        df["en01"]
        .astype(str)
        .str.strip()
        .map({
            "Yes": 1,
            "No": 0
        })
    )


if "en01b" in df.columns:

    df["business_closure_count"] = pd.to_numeric(
        df["en01b"],
        errors="coerce"
    )


# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFinancial feature engineering completed.")
print("Output:", OUTPUT_FILE)
print("Final shape:", df.shape)