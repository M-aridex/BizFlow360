from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "processed"
    / "msme_financial_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# Load feature-engineered data
# ---------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Input dataset shape:", df.shape)


# ---------------------------------------------------------------
# Candidate financial/modeling features
# ---------------------------------------------------------------

candidate_features = [
    "eh01_1_clean",
    "eh15_1_clean",
    "eh04_1_clean",
    "net_income_margin",
    "revenue_change_ratio",
    "business_closed",
    "business_closure_count"
]


# Keep only columns that actually exist
model_features = [
    column
    for column in candidate_features
    if column in df.columns
]


print("\nModel features:")

for feature in model_features:
    print(" -", feature)


# ---------------------------------------------------------------
# Create modeling dataset
# ---------------------------------------------------------------

model_data = df[model_features].copy()


# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------

output_file = (
    OUTPUT_DIR
    / "msme_financial_model_dataset.csv"
)

model_data.to_csv(
    output_file,
    index=False
)

print("\nModel dataset created.")
print("Output:", output_file)
print("Shape:", model_data.shape)