from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "raw" / "kaggle"
OUTPUT_DIR = PROJECT_ROOT / "processed" / "world_bank"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# Locate World Bank CSV
# ---------------------------------------------------------------

csv_files = list(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No World Bank CSV file was found in the Kaggle raw directory."
    )

print("World Bank files found:")

for file in csv_files:
    print(" -", file.name)


# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------

data_file = csv_files[0]

print("\nLoading:", data_file)

world_bank = pd.read_csv(
    data_file,
    low_memory=False
)

print("Original shape:", world_bank.shape)


# ---------------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------------

world_bank.columns = (
    world_bank.columns
    .str.strip()
)

world_bank = world_bank.dropna(
    axis=1,
    how="all"
)


# ---------------------------------------------------------------
# Save cleaned World Bank dataset
# ---------------------------------------------------------------

output_file = OUTPUT_DIR / "world_bank_cleaned.csv"

world_bank.to_csv(
    output_file,
    index=False
)

print("Saved:", output_file)
print("Final shape:", world_bank.shape)