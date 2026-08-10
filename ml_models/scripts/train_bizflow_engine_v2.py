"""
BizFlow360 — Engine v2.0: leakage audit, calibration & full validation.
Addresses review points: (1) calibration, (2) target leakage, (10) proper validation.

Run from anywhere:  python ml_models/scripts/train_bizflow_engine_v2.py

Outputs:
  ml_models/models/trained/on_real_data/bizflow_engine_v2.0.joblib      (calibrated production engine)
  ml_models/models/trained/on_real_data/bizflow_engine_v2.0_base.joblib (uncalibrated, for SHAP)
  ml_models/models/trained/on_real_data/model_card_v2.json              (metrics + provenance + meanings)
  ml_models/models/metrics/on_real_data/validation_metrics_v2.csv
  ml_models/models/metrics/on_real_data/confusion_matrix_v2.png
  ml_models/models/metrics/on_real_data/calibration_curve_v2.png
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss, confusion_matrix,
                             ConfusionMatrixDisplay)
from lightgbm import LGBMClassifier

# ---------------------------------- paths ----------------------------------
BASE = Path(__file__).resolve().parents[2]                      # BizFlow360/
DATA = BASE / "ml_models" / "data" / "unified_msme_modeling_data.csv"
TRAIN_DIR = BASE / "ml_models" / "models" / "trained" / "on_real_data"
METRIC_DIR = BASE / "ml_models" / "models" / "metrics" / "on_real_data"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "distress_label"
CAT_COLS = ["county", "sector"]

# Leakage-safe numeric features (profitability-derived features EXCLUDED)
SAFE_NUM = [
    "male_working_owners", "female_working_owners",
    "total_monthly_expenses", "monthly_rent_expense", "monthly_electricity_expense",
    "monthly_credit_expense", "monthly_social_responsibility_expense",
    "revenue_last_month", "normal_monthly_revenue",
    "stock_value_beginning", "stock_value_end", "total_turnover_2015",
    "revenue_change_ratio", "business_closed", "number_closed_establishments",
    "revenue_decline", "low_revenue",
]
# Excluded from the production model (audit only): they overlap definitionally
# with the self-reported performance target. They remain in the app ONLY as
# inputs to the advice engine, never to the predictor.
EXCLUDED = ["net_income_last_month", "net_income_margin", "zero_or_missing_net_income"]

# ------------------------------ load & clean -------------------------------
df = pd.read_csv(DATA)

# KNBS "missing" placeholders appear as impossible negative numbers -> NaN
for col in SAFE_NUM + EXCLUDED:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].mask(df[col] < 0, np.nan)   # negatives are placeholders here
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------- pipeline ----------------------------------
def build_pipeline(num_cols):
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                 unknown_value=-1))]), CAT_COLS),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
    ])
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                         num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    return Pipeline([("pre", pre), ("clf", clf)])

def evaluate(proba, y_true, tag):
    pred = (proba >= 0.5).astype(int)
    return {
        "set": tag,
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "brier": round(brier_score_loss(y_true, proba), 4),
    }

results = []

# --- (2) LEAKAGE AUDIT: FULL vs LEAKAGE_SAFE (uncalibrated, held-out test) ---
full_pipe = build_pipeline(SAFE_NUM + EXCLUDED)
full_pipe.fit(X_train, y_train)
proba_full = full_pipe.predict_proba(X_test)[:, 1]
results.append(evaluate(proba_full, y_test, "FULL_incl_profitability"))

safe_pipe = build_pipeline(SAFE_NUM)
safe_pipe.fit(X_train, y_train)
proba_safe = safe_pipe.predict_proba(X_test)[:, 1]
results.append(evaluate(proba_safe, y_test, "LEAKAGE_SAFE"))

# --- (1)+(10) CALIBRATION: Platt scaling on the leakage-safe engine ---
calibrated = CalibratedClassifierCV(build_pipeline(SAFE_NUM), cv=5, method="sigmoid")
calibrated.fit(X_train, y_train)
proba_cal = calibrated.predict_proba(X_test)[:, 1]
results.append(evaluate(proba_cal, y_test, "LEAKAGE_SAFE_CALIBRATED (production)"))

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(METRIC_DIR / "validation_metrics_v2.csv", index=False)
print(metrics_df.to_string(index=False))

# ------------------------------ plots --------------------------------------
cm = confusion_matrix(y_test, (proba_cal >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Stable", "Distressed"]).plot(
    ax=ax, cmap="Blues", colorbar=True)
ax.set_title("BizFlow Engine v2.0 — Confusion Matrix (held-out test)")
fig.tight_layout()
fig.savefig(METRIC_DIR / "confusion_matrix_v2.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
for proba, name, c in [(proba_safe, "uncalibrated", "#94a3b8"),
                       (proba_cal, "calibrated (sigmoid)", "#0f766e")]:
    frac, mean = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
    ax.plot(mean, frac, marker="o", label=name, color=c)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Calibration curve — BizFlow Engine v2.0")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(METRIC_DIR / "calibration_curve_v2.png", dpi=200)
plt.close(fig)

# ------------------------------ save artifacts -----------------------------
joblib.dump(calibrated, TRAIN_DIR / "bizflow_engine_v2.0.joblib")
joblib.dump(safe_pipe, TRAIN_DIR / "bizflow_engine_v2.0_base.joblib")

card = {
    "version": "2.0",
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "model": "LightGBM, Platt-sigmoid calibrated (5-fold)",
    "label_provenance": ("Target = KNBS question eh09, owner self-reported performance "
                         "'Bad'. The label is NOT computed from any financial feature, "
                         "so there is no direct definitional leakage."),
    "leakage_policy": ("Profitability-derived features (net_income_last_month, "
                       "net_income_margin, zero_or_missing_net_income) are EXCLUDED from "
                       "the predictor. They are used only by the advice engine in the app."),
    "production_feature_set": "LEAKAGE_SAFE (19 features)",
    "metrics": results,
    "feature_meanings": {
        "County": "Typical distress rate of businesses in this county in the KNBS data.",
        "Sector": "How businesses in this sector performed nationally.",
        "Male Owners": "Number of active male owners (management depth).",
        "Female Owners": "Number of active female owners (management depth).",
        "Total Monthly Expenses": "Your monthly cost base.",
        "Monthly Rent": "Rent component of your costs.",
        "Monthly Electricity": "Electricity component of your costs.",
        "Monthly Credit Payments": "Loan/credit repayments inside your costs.",
        "Monthly Social Responsibility": "Community/social contributions inside your costs.",
        "Revenue Last Month": "Actual sales last month.",
        "Normal Monthly Revenue": "Your typical monthly sales.",
        "Stock (Beginning)": "Inventory value at month start.",
        "Stock (End)": "Inventory value at month end (sales velocity signal).",
        "Annual Turnover": "Yearly business size benchmark.",
        "Revenue Change Ratio": "Last month's revenue divided by normal revenue (trend).",
        "Closed Any Establishment": "Whether you closed a branch in the last 5 years.",
        "Number Closed": "How many branches were closed.",
        "Revenue Declined": "Your own confirmation that revenue is falling.",
        "Low Revenue Flag": "Micro-scale business flag (revenue below KES 10,000).",
    },
}
with open(TRAIN_DIR / "model_card_v2.json", "w") as f:
    json.dump(card, f, indent=2)

print("\n✅ Engine v2.0 artifacts saved.")