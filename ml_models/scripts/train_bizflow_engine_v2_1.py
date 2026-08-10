"""
BizFlow360 Engine v2.1 — robust features + tuning + threshold selection + CIs.
Run: python ml_models/scripts/train_bizflow_engine_v2_1.py
Outputs v2.1 artifacts (engine, base engine, model card, plots).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from lightgbm import LGBMClassifier

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "ml_models" / "data" / "unified_msme_modeling_data.csv"
TRAIN_DIR = BASE / "ml_models" / "models" / "trained" / "on_real_data"
METRIC_DIR = BASE / "ml_models" / "models" / "metrics" / "on_real_data"
TRAIN_DIR.mkdir(parents=True, exist_ok=True); METRIC_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "distress_label"
CATS = ["county", "sector"]
NUM = [
    "male_working_owners", "female_working_owners",
    "total_monthly_expenses", "monthly_rent_expense", "monthly_electricity_expense",
    "monthly_credit_expense", "monthly_social_responsibility_expense",
    "revenue_last_month", "normal_monthly_revenue",
    "stock_value_beginning", "stock_value_end", "total_turnover_2015",
    "revenue_change_ratio", "business_closed", "number_closed_establishments",
    "revenue_decline", "low_revenue",
]
EXCLUDED = ["net_income_last_month", "net_income_margin", "zero_or_missing_net_income"]

# ---------------- load + robust cleaning ----------------
df = pd.read_csv(DATA)
for col in NUM + EXCLUDED:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].mask(df[col] < 0, np.nan)          # KNBS placeholders

df["revenue_change_ratio"] = df["revenue_change_ratio"].clip(0, 20)          # tame 8000x tails
df["expense_to_revenue"] = (df["total_monthly_expenses"] / df["revenue_last_month"].clip(lower=1)).clip(0, 10)
df["stock_change"] = ((df["stock_value_end"] - df["stock_value_beginning"])
                      / df["stock_value_beginning"].clip(lower=1)).clip(-1, 5)
df["turnover_missing"] = (df["total_turnover_2015"].isna() | (df["total_turnover_2015"] == 288000)).astype(float)
df["revenue_missing"] = df["revenue_last_month"].isna().astype(float)
NUM_V21 = NUM + ["expense_to_revenue", "stock_change", "turnover_missing", "revenue_missing"]

df = df.dropna(subset=[TARGET])
X = df[CATS + NUM_V21]; y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------- pipeline ----------------
def build_pipeline(params):
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CATS),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_V21),
    ])
    return Pipeline([("pre", pre), ("clf", LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params))])

def cv_auc(pipe, X, y, cv=3):
    skf = StratifiedKFold(cv, shuffle=True, random_state=42)
    out = []
    for tr, va in skf.split(X, y):
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        out.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    return float(np.mean(out))

# ---------------- small hyperparameter search ----------------
GRID = [
    dict(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31, scale_pos_weight=1.0),
    dict(n_estimators=500, learning_rate=0.03, max_depth=7, num_leaves=63, scale_pos_weight=1.0),
    dict(n_estimators=500, learning_rate=0.05, max_depth=5, num_leaves=31, scale_pos_weight=1.74),
    dict(n_estimators=400, learning_rate=0.05, max_depth=6, num_leaves=63, scale_pos_weight=1.74),
    dict(n_estimators=800, learning_rate=0.02, max_depth=8, num_leaves=127, scale_pos_weight=1.0),
]
scores = [(p["n_estimators"], p["scale_pos_weight"], cv_auc(build_pipeline(p), X_train, y_train)) for p in GRID]
print("CV search (n_estimators, spw, AUC):", scores)
best_params = GRID[int(np.argmax([s[2] for s in scores]))]

# ---------------- threshold selection on a held-out val split ----------------
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                            random_state=7, stratify=y_train)
cal_val = CalibratedClassifierCV(build_pipeline(best_params), cv=5, method="sigmoid").fit(X_tr, y_tr)
val_prob = cal_val.predict_proba(X_val)[:, 1]
thr_grid = np.linspace(0.05, 0.90, 171)
f1s = [f1_score(y_val, (val_prob >= t).astype(int), zero_division=0) for t in thr_grid]
tau_f1 = float(thr_grid[int(np.argmax(f1s))])
recalls = [recall_score(y_val, (val_prob >= t).astype(int), zero_division=0) for t in thr_grid]
tau_r60 = float(max([t for t, r in zip(thr_grid, recalls) if r >= 0.60], default=tau_f1))

# ---------------- final model: refit on full train, calibrate, evaluate ----------------
cal = CalibratedClassifierCV(build_pipeline(best_params), cv=5, method="sigmoid").fit(X_train, y_train)
base = build_pipeline(best_params).fit(X_train, y_train)
prob = cal.predict_proba(X_test)[:, 1]
pred = (prob >= tau_f1).astype(int)

metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred, zero_division=0), 4),
    "recall": round(recall_score(y_test, pred, zero_division=0), 4),
    "f1": round(f1_score(y_test, pred, zero_division=0), 4),
    "roc_auc": round(roc_auc_score(y_test, prob), 4),
    "pr_auc": round(average_precision_score(y_test, prob), 4),
    "brier": round(brier_score_loss(y_test, prob), 4),
}

# bootstrap CIs
rng = np.random.default_rng(7); aucs, briers = [], []
for _ in range(250):
    idx = rng.integers(0, len(y_test), len(y_test))
    if len(np.unique(y_test.iloc[idx])) < 2: continue
    aucs.append(roc_auc_score(y_test.iloc[idx], prob[idx]))
    briers.append(brier_score_loss(y_test.iloc[idx], prob[idx]))
metrics["roc_auc_ci95"] = [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]
metrics["brier_ci95"] = [round(float(np.percentile(briers, 2.5)), 4), round(float(np.percentile(briers, 97.5)), 4)]
print(json.dumps(metrics, indent=2))

# ---------------- plots ----------------
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Stable", "Distressed"]).plot(ax=ax, cmap="Blues")
ax.set_title(f"Engine v2.1 @ tau={tau_f1:.2f}")
fig.tight_layout(); fig.savefig(METRIC_DIR / "confusion_matrix_v2.1.png", dpi=200); plt.close(fig)

frac, mean = calibration_curve(y_test, prob, n_bins=10, strategy="uniform")
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(mean, frac, "o-", color="#0f766e", label="v2.1 calibrated")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("Calibration curve — Engine v2.1"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(METRIC_DIR / "calibration_curve_v2.1.png", dpi=200); plt.close(fig)

# ---------------- save ----------------
joblib.dump(cal, TRAIN_DIR / "bizflow_engine_v2.1.joblib")
joblib.dump(base, TRAIN_DIR / "bizflow_engine_v2.1_base.joblib")

card = {
    "version": "2.1", "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "model": "LightGBM (tuned) + sigmoid calibration",
    "best_params": best_params,
    "operating_threshold_max_f1": tau_f1,
    "operating_threshold_recall60": tau_r60,
    "metrics_test": metrics,
    "features": CATS + NUM_V21,
    "engineered_features": {
        "revenue_change_ratio": "clip(last/normal, 0, 20)",
        "expense_to_revenue": "clip(expenses/max(revenue,1), 0, 10)",
        "stock_change": "clip((end-begin)/max(begin,1), -1, 5)",
        "turnover_missing": "1 if turnover missing or == 288000 placeholder",
        "revenue_missing": "1 if revenue missing",
    },
    "leakage_policy": "Profitability features excluded from predictor (v2 audit: delta-AUC 0.012).",
}
(TRAIN_DIR / "model_card_v2.1.json").write_text(json.dumps(card, indent=2))
print("✅ Engine v2.1 artifacts saved.")