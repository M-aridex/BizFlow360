from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_MSME = (
    PROJECT_ROOT
    / "raw"
    / "knbs"
    / "MSME_final_establishments_dataset_with_trainings.dta"
)

OUTPUT_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------

print("Loading KNBS MSME dataset...")

msme = pd.read_stata(RAW_MSME)

print("Original shape:", msme.shape)


# ---------------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------------

msme.columns = (
    msme.columns
    .str.strip()
    .str.lower()
)

# Remove completely empty columns
msme = msme.dropna(axis=1, how="all")

print("Shape after basic cleaning:", msme.shape)


# ---------------------------------------------------------------
# Save cleaned MSME dataset
# ---------------------------------------------------------------

output_file = OUTPUT_DIR / "msme_analysis_cleaned.csv"

msme.to_csv(output_file, index=False)

print("Saved:", output_file)
print("Final shape:", msme.shape)