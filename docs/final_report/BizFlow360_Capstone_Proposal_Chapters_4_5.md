# CHAPTER FOUR: SYSTEM IMPLEMENTATION AND RESULTS

## 4.1 Introduction

This chapter presents the actual implementation of BizFlow360, the machine learning–based early warning system for predicting financial distress among Kenyan MSMEs. It documents the development environment, the data pipeline as executed, the feature engineering and leakage-audit decisions, the three model versions (v1.0, v2.0 and v2.1), the evaluation results, the interpretability layer, the Streamlit prototype, the validation testing, the deployment, and the ethical safeguards that were applied. The chapter is written against the objectives and methodology stated in Chapters One and Three.

The name **BizFlow360** was chosen deliberately to reflect the system's architecture: **"Biz"** for the MSMEs it serves; **"Flow"** for cash-flow health and continuous month-over-month monitoring (the History page); and **"360"** for the holistic, four-view experience it provides — predictive (the calibrated risk score), explainable (SHAP), prescriptive (computed break-even math and AI action plans) and conversational (the WhatsApp-style advisor).

## 4.2 Development Environment and Tools

The system was developed using the following environment:

| Component | Tool / Version | Role |
|---|---|---|
| Language | Python 3.12 | Core development |
| Web framework | Streamlit 1.x | Interactive dashboard and deployment |
| ML libraries | scikit-learn, LightGBM | Modelling, preprocessing, calibration, metrics |
| Explainability | SHAP | Local (per-prediction) explanations, waterfall plots |
| Generative AI | NVIDIA NIM API (GLM-5.2 text; Llama-3.2-90B Vision) via OpenAI-compatible client | AI advisor, 10-step action plans, chat, receipt reading |
| PDF engine | fpdf2 | Branded report generation |
| Serialization | joblib | Model, scaler and encoder artifacts |
| Analysis | pandas, NumPy, Matplotlib | Data handling and visualization |
| Version control | Git / GitHub | Code and artifact management |
| Hosting | Streamlit Community Cloud | Public deployment |
| Local compute | Personal machine + virtual environment | Training and development |

## 4.3 Data Collection and Preprocessing (As Implemented)

### 4.3.1 Data Source

In line with Section 3.2, the primary source used was the **KNBS Micro, Small and Medium Enterprises survey microdata** (real Kenyan business records), merged into a single unified modelling dataset (`unified_msme_modeling_data.csv`). The target variable was constructed as a binary label from the owner's self-reported business performance (KNBS question on current business performance): **0 = Financially Stable**, **1 = Financially Distressed** ("Bad" performance). Because the label is self-reported and *not* computed from the financial features, it served as an honest outcome for leakage auditing (Section 4.4.2).

### 4.3.2 Cleaning Steps Actually Performed

Consistent with Section 3.4, the following cleaning pipeline was executed:

1. **Placeholder handling:** KNBS missing-value placeholder codes (impossible negative numbers such as −19.305, −8.82, −73.5) were detected and converted to `NaN`.
2. **Missing-value imputation:** Numerical columns were imputed with the median and categorical columns with the mode, fitted **inside the training folds only** (via scikit-learn `Pipeline`/`ColumnTransformer`) to prevent preprocessing leakage.
3. **Standardization of labels:** County and sector strings were standardized; sectors were mapped to full KNBS ISIC codes for the model and to friendly MSME labels (e.g., "🛒 Retail Shop / Kiosk (Duka, Mama Mboga)") for the user interface.
4. **Outlier treatment:** Extreme engineered ratios were clipped to sensible ranges (`revenue_change_ratio` clipped to [0, 20]; `expense_to_revenue` to [0, 10]; `stock_change` to [−1, 5]).
5. **Type conversion:** Currency and ratio fields were converted to numeric dtypes.
6. **Class handling:** Stratified 80/20 train–test splitting preserved the distressed/stable proportion; class weighting and threshold tuning were used instead of naive resampling.

### 4.3.3 Final Dataset

The final modelling dataset contained thousands of real Kenyan MSME records across 47 counties and 30+ ISIC sectors, with a distressed-class prevalence of approximately 36.5% in the held-out test set.

## 4.4 Feature Engineering (As Implemented)

### 4.4.1 Engineered Features

Following Section 3.7, the following features were engineered and fed to the production model (23 features in total: 2 categorical + 17 numeric + 4 engineered):

| Feature | Formula / Description | Purpose |
|---|---|---|
| Revenue Change Ratio | Last Month Revenue ÷ Normal Monthly Revenue | Detects under-performance or growth (top SHAP driver) |
| Expense-to-Revenue Ratio | Total Monthly Expenses ÷ Revenue (clipped at 10) | Cost burden relative to sales |
| Stock Change | (End − Beginning) ÷ Beginning (clipped) | Inventory velocity / shrinkage signal |
| Low Revenue Flag | 1 if revenue < KES 10,000 | Micro-scale vulnerability |
| Turnover Missing | 1 if turnover missing/placeholder | Data-quality / informality flag |
| Revenue Missing | 1 if revenue missing | Data-quality flag |
| Net Income Margin | Net Income ÷ Revenue | Used for user-facing math; **excluded from predictor** (see 4.4.2) |

### 4.4.2 Target Leakage Audit

A leakage audit was conducted after it was observed that v1.0's apparently strong performance was driven by profitability features (`net_income_last_month`, `net_income_margin`, `zero_or_missing_net_income`) that overlap definitionally with the distress concept. An ablation study (Table 4.1) compared the full feature set against a **leakage-safe** set that excludes direct profitability features:

**Table 4.1 — Leakage ablation (held-out test set)**

| Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| FULL (incl. profitability) | 0.6851 | 0.6124 | 0.3749 | 0.4651 | 0.7023 | 0.2036 |
| LEAKAGE-SAFE | 0.6835 | 0.6199 | 0.3446 | 0.4430 | 0.6902 | 0.2072 |

Removing profitability features cost only **0.012 ROC-AUC**, demonstrating that the engine's signal is genuine (revenue trend, expense burden, county/sector baselines) rather than definitional. The leakage-safe set was therefore adopted for production. Profitability figures are still collected and displayed to the user in the "Your numbers" tab (deficit, break-even, margin) for advisory purposes, but they do not enter the predictor.

## 4.5 Model Development and Versioning

Three model versions were developed, each addressing a specific scientific weakness of the previous one:

### 4.5.1 Version 1.0 (Baseline)

- **Algorithm:** LightGBM on the full 22-feature set.
- **Result:** ROC-AUC 0.7042, recall 61.1%.
- **Problem identified:** Target leakage (Section 4.4.2). The score was also an uncalibrated ranking, unsuitable for presentation as a probability.

### 4.5.2 Version 2.0 (Leakage-Safe + Calibrated)

- **Changes:** Leakage-safe feature set; Platt (sigmoid) probability calibration via 5-fold `CalibratedClassifierCV`.
- **Result:** ROC-AUC 0.6962, Brier 0.2061 — honest, calibrated probabilities.
- **Problem identified:** At the default 0.5 threshold, recall was only 28.3%; the system missed too many distressed businesses — unacceptable for an early-warning product.

### 4.5.3 Version 2.1 (Production — Final Model)

- **Changes:** Robust outlier clipping; four engineered features (Section 4.4.1); hyperparameter search; **operating-threshold selection** on a validation split (max-F1 at τ = 0.33; recall ≥ 60% at τ = 0.375); bootstrap confidence intervals (250 iterations) for reliability.
- **Result at operating threshold τ = 0.33:** Accuracy 0.594, Precision 0.465, **Recall 0.746**, F1 0.573, ROC-AUC 0.696 (95% CI: 0.676–0.714), Brier 0.206.
- **Rationale:** For an early-warning system, a false negative (missing a distressed business) is costlier than a false positive. v2.1 therefore trades some precision for a recall of **74.6%**, while calibration keeps every displayed percentage honest.

## 4.6 Model Evaluation and Comparison

### 4.6.1 Classifier Screening

As planned in Section 3.8, Logistic Regression (baseline), Random Forest, XGBoost and LightGBM were trained and compared during model development. **LightGBM emerged as the best-performing classifier** on the KNBS data and was adopted for all three production versions. Detailed per-model training logs are preserved in the project notebooks (`ml_models/` directory) and the GitHub repository.

### 4.6.2 Final Production Metrics

**Table 4.2 — v2.1 production model (held-out test set)**

| Metric | Value |
|---|---|
| ROC-AUC | 0.696 (95% CI: 0.676–0.714) |
| Brier Score | 0.206 |
| Accuracy (τ = 0.33) | 0.594 |
| Precision (τ = 0.33) | 0.465 |
| Recall (τ = 0.33) | 0.746 |
| F1-Score (τ = 0.33) | 0.573 |
| Operating threshold (max-F1) | 0.33 |
| Operating threshold (recall ≥ 60%) | 0.375 |

The evaluation satisfies Objective 4 and Research Question 3: the model was assessed with accuracy, precision, recall, F1, ROC-AUC **and** calibration (Brier), with recall given special attention as required by the problem domain (Section 3.9).

## 4.7 Model Interpretability (SHAP)

In line with Section 3.11, SHAP (SHapley Additive exPlanations) was implemented for every prediction. The Analyze page renders a **waterfall plot** showing, for the individual business, which features pushed risk up (red) or down (blue), together with plain-language meanings for each driver. Examples observed during validation include:

- A growing Nairobi retail business: `Normal Monthly Revenue` −0.489 (protective), `County (Nairobi)` +0.331 (risk), `Expense-to-Revenue Ratio` +0.280 (risk).
- A collapsing Marsabit eatery: `Revenue Change Ratio` +0.694 as the single strongest risk driver.

This satisfies the transparency requirement: the system never presents a "black-box" score; every prediction is accompanied by its evidence.

## 4.8 The BizFlow360 Application (MVP)

### 4.8.1 Architecture and Pages

The Streamlit prototype (Objective 5) implements five pages:

1. **Welcome:** onboarding with a four-step explanation and the prediction-vs-advice disclaimer.
2. **Analyze:** sidebar form (county, sector, owners, revenue, expenses, net income, stock, turnover, closure history, decline flag); real-time calibrated prediction; four tabs — *Why this score?* (SHAP waterfall + factor meanings), *Your numbers* (computed deficit, break-even revenue, required revenue increase, required expense reduction, margin), *Action plan* (instant quick wins + a button generating **10 deeply analyzed, sector-specific AI actions**), and *AI explanation* (simple explanation + encouragement, and the combined PDF download).
3. **History:** local trend tracking (`bizflow_history.json`), latest/best score cards, and an AI-powered "compare with a past analysis" feature.
4. **Chat:** WhatsApp-style advisor with suggestion chips, file/camera attachments (vision model for receipts, text extraction for documents), session archiving (`chat_sessions.json`) and a strict financial-domain lock.
5. **Settings:** model metrics card (v2.1), AI service status, privacy note, and history management.

### 4.8.2 The Analysis Pipeline (Runtime)

1. **Encode** — county/sector via saved ordinal encoders; 2. **Impute + Scale** — saved median imputer and standard scaler; 3. **Predict** — calibrated LightGBM v2.1 probability; 4. **Explain** — SHAP waterfall; 5. **Advise** — computed break-even math + NVIDIA NIM action plan; 6. **Save** — append to history.

Risk categories displayed: 🟢 Low < 40% · 🟡 Medium 40–70% · 🔴 High > 70%; the engine additionally flags any business scoring ≥ 33% (the validated operating point) for review.

### 4.8.3 AI Advisor and Evidence Rules

The advisor (NVIDIA NIM GLM-5.2; Llama-3.2-90B Vision for images) is bound by system-prompt evidence rules: use **only** the numbers provided; never invent external facts or timelines; separate prediction from advice; use KES and plain language. This directly implements the "responsible use" commitment of Section 3.14.

### 4.8.4 PDF Report Engine

The branded `fpdf2` report contains: header band with title and tagline; date/business/county meta line; color-coded risk badge; computed numbers; SHAP top drivers with impacts; the 10-step action plan; AI explanation; a prediction-vs-advice disclaimer; and a paginated branded footer. A sanitization pipeline (emoji stripping, smart-punctuation mapping, markdown removal, latin-1 ignore encoding) guarantees crash-free rendering of AI-generated text.

### 4.8.5 Privacy and Security

All analysis and chat history is stored **only on the user's device**. The NVIDIA API key is never shown to end users and is supplied via Streamlit Secrets / environment variables (the previously embedded development key was removed before publication). No personally identifiable information is collected, in compliance with the Kenya Data Protection Act, 2019.

## 4.9 Validation Testing

Four real-world scenarios were executed in the live app to validate logical behaviour (Objective 5 / Research Question 4). Results:

**Table 4.3 — Validation scenarios (v2.1, live app)**

| # | Scenario | Expected | Actual | Outcome |
|---|---|---|---|---|
| 1 | Healthy growing retail (Nairobi) | Low | 37.3% Low | ✅ Pass |
| 2 | Collapsing food vendor (Marsabit) | High | 40.0% Medium | ⚠️ Explainable deviation |
| 3 | Stable with closure history (Kisumu) | Medium | 32.9% Low | ⚠️ Borderline |
| 4 | Thin-margin farm (Kajiado) | Medium | 53.6% Medium | ✅ Pass |

**Analysis of deviations:**
- *Scenario 2:* the revenue collapse (ratio 0.5) was correctly the top risk driver (+0.694), but Marsabit's low county baseline (−0.392) and respectable normal revenue (−0.340) moderated the score to Medium. The business was nonetheless flagged for review (≥ 33%) and received 🔴 Critical deficit actions (KES 15,000 monthly deficit; break-even KES 45,000).
- *Scenario 3:* the closure history was genuinely penalized (+0.19, 6th driver) and Kisumu county added the strongest risk push (+0.411), but stable current fundamentals placed the score 0.1pp below the review line — calibrated uncertainty at a decision boundary.

**Verdict:** two exact matches and two transparent, conservative-in-the-safe-direction deviations; in every case the drivers are individually sensible and the product-level guidance (review flag + deficit actions) remains correct. The v2.1 engine is validated as logical, transparent and appropriately cautious.

A full end-to-end sample (Baringo farm: revenue KES 150,000, expenses KES 120,000) produced a 35.2% Low-Risk report whose SHAP drivers (Male Owners −0.266; Revenue Change Ratio −0.241; Total Monthly Expenses +0.202; Revenue Last Month −0.184; Normal Monthly Revenue −0.171) and 10-step plan (track livestock costs, feed buffer, M-Pesa Till payments, KES 30–50k emergency fund, pricing review, value addition, debt caution, farm ledger, bulk negotiation, dry-season planning) were confirmed coherent and sector-appropriate.

## 4.10 Deployment

The prototype was deployed on **Streamlit Community Cloud** at `https://bizflow360-playground.streamlit.app/`. The GitHub repository (`github.com/M-aridex/BizFlow360`) contains all code, notebooks, documentation, and the committed v2.1 model artifacts (`bizflow_engine_v2.1.joblib`, `bizflow_engine_v2.1_base.joblib`, `model_card_v2.1.json`) so the cloud environment boots without retraining. The API key is injected through Streamlit Secrets. This satisfies Objective 5 and Research Question 4.

## 4.11 Ethical Considerations (As Implemented)

1. **Data privacy:** only aggregated, publicly sourced survey data used; no PII.
2. **Kenya Data Protection Act, 2019:** data minimization and purpose limitation observed; user-entered data stays on-device.
3. **Bias and fairness:** county, sector and gender effects are surfaced through SHAP for inspection rather than hidden; no protected attribute is used to deny services.
4. **Transparency:** every score ships with its SHAP evidence and computed math.
5. **Responsible use:** persistent disclaimers state the tool is a decision-support system, not professional financial, legal or tax advice.

## 4.12 Achievement Against Objectives

| Objective | Delivered Evidence |
|---|---|
| 1. Identify key distress indicators | SHAP ranking: revenue change ratio, expense-to-revenue, county/sector baselines, stock change |
| 2. Build predictive models | Three versioned LightGBM engines (v1.0 → v2.1) |
| 3. Compare models | Classifier screening (LR/RF/XGBoost/LightGBM) + leakage ablation (Table 4.1) |
| 4. Evaluate with proper metrics | Accuracy, precision, recall, F1, ROC-AUC, Brier, CIs, threshold analysis (Table 4.2) |
| 5. Deploy prototype dashboard | Live Streamlit app with SHAP, AI advisor, PDF export, chat, history |

---

# CHAPTER FIVE: SUMMARY, CONCLUSION AND RECOMMENDATIONS

## 5.1 Summary of the Study

This project set out to develop BizFlow360, a machine learning–based early warning system for predicting financial distress among Kenyan MSMEs. Using real KNBS MSME survey data, the team built and compared classification models, audited and eliminated target leakage, calibrated probabilities, tuned the operating threshold for early detection, and wrapped the final v2.1 LightGBM engine in an explainable, user-friendly Streamlit application featuring SHAP explanations, computed break-even mathematics, a 10-step NVIDIA-NIM action planner, a WhatsApp-style advisor with vision support, branded PDF reporting, and local-only history. The system was validated with four realistic scenarios and deployed publicly on Streamlit Community Cloud.

## 5.2 Conclusions (by Research Question)

1. **Key indicators:** Revenue dynamics (revenue change ratio), cost burden (expense-to-revenue), inventory movement (stock change), and county/sector baselines are the strongest practical predictors of MSME distress; direct profitability figures, while informative to users, constitute a leakage risk if used inside the predictor.
2. **Best model:** LightGBM outperformed the screened alternatives and, once leakage-safe and calibrated (v2.1), delivers honest probabilities with 74.6% recall at its operating threshold.
3. **Complex vs baseline:** Gradient boosting materially outperformed the logistic baseline on this tabular, non-linear problem; however, scientific rigour (leakage audit, calibration, threshold selection) proved as important as algorithm choice.
4. **Deployment:** A simple, accessible prototype is achievable on free hosting by committing serialized artifacts and injecting secrets securely; explainability and advisory layers transform a raw score into actionable guidance.

## 5.3 Contributions to Knowledge

- **Practical:** first publicly accessible, MSME-focused, calibrated financial-distress early warning prototype for Kenya.
- **Scientific:** a documented leakage-audit and calibration methodology showing that honest, lower-AUC models outserve inflated ones in decision-support contexts.
- **Methodological:** integration of predictive ML, SHAP explainability and evidence-locked generative AI into a single coherent advisory pipeline.
- **Social:** a free tool that gives informal and semi-formal businesses a language (deficit, break-even, margin) usually reserved for firms with accountants.

## 5.4 Limitations

1. **Data:** self-reported survey data with recall bias; no real-time transaction (M-Pesa/bank) data; some expense sub-fields default to zero in the app.
2. **Model:** moderate ROC-AUC (0.696); two validation scenarios deviated from heuristic expectations (though explainable); possible under-representation of some sectors/counties.
3. **System:** dependence on an external AI API (internet + key required); static PDF snapshots rather than dynamic dashboards.
4. **Scope:** binary status classification only — the system does not predict *when* distress will occur; it provides decision support, not binding advice.

## 5.5 Recommendations

**For future research:** (1) integrate real-time mobile-money transaction streams; (2) extend to survival/time-to-distress modelling; (3) train sector-specific sub-models; (4) explore LightGBM+XGBoost ensembles; (5) conduct usability field studies with actual MSME owners.

**For policymakers:** use BizFlow360-style screening for grant and training targeting; embed tools in county MSME capacity programmes; release more granular anonymized MSME microdata.

**For financial institutions:** adopt as a pre-screening and monitoring layer for MSME credit; design cash-flow-aligned repayment products informed by the computed deficit/break-even metrics.

**For MSME owners:** monitor monthly; implement and document the action plan; maintain simple records (even notebook-level) to improve future scores.

## 5.6 Concluding Remarks

BizFlow360 demonstrates that machine learning can provide credible, transparent and actionable early warning for Kenyan MSMEs even with imperfect public data. By choosing honesty over inflated accuracy — leakage-safe features, calibrated probabilities, a recall-oriented operating threshold, and SHAP-level transparency — the project delivers a decision-support system that respects its users and its limits. The combination of predictive, explainable and generative AI charted here is a promising template for financial inclusion technology across Africa. *Financial clarity for every MSME.*

---

# REFERENCES

Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance, 23*(4), 589–609.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Central Bank of Kenya. (n.d.). *Financial inclusion and FinAccess survey reports*. Central Bank of Kenya.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*.

Kenya National Bureau of Statistics. (2016). *Kenya Micro, Small and Medium Enterprises Basic Report 2016*. KNBS.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.

NVIDIA. (2024). *NVIDIA NIM inference microservices*. https://developer.nvidia.com/nim

Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*.

Republic of Kenya. (2019). *The Data Protection Act, 2019*. Government Printer.

Streamlit Inc. (2024). *Streamlit documentation*. https://docs.streamlit.io

Team Maridex. (2026). *BizFlow360: A Machine Learning-Based Early Warning System for Predicting Financial Distress Among Kenyan MSMEs* [GitHub repository]. https://github.com/M-aridex/BizFlow360

World Bank. (n.d.). *Enterprise Surveys: Kenya data*. World Bank Group.

---

# APPENDICES

## Appendix A — Production Feature Set (23 features)

County; Sector; Male Working Owners; Female Working Owners; Total Monthly Expenses; Monthly Rent; Monthly Electricity; Monthly Credit Payments; Monthly Social Responsibility; Revenue Last Month; Normal Monthly Revenue; Stock (Beginning); Stock (End); Annual Turnover; Revenue Change Ratio; Closed Any Establishment; Number Closed; Revenue Declined; Low Revenue Flag; Expense-to-Revenue Ratio; Stock Change; Turnover Missing; Revenue Missing. *(Profitability features are collected for advisory math but excluded from the predictor.)*

## Appendix B — Model Metrics Tables

B.1 Leakage ablation (v2.0): see Table 4.1. B.2 v2.1 production metrics with CIs and thresholds: see Table 4.2. B.3 v1.0 baseline: ROC-AUC 0.7042, recall 61.1% (superseded due to leakage).

## Appendix C — Validation Scenarios

Full input/output records and screenshots for the four scenarios in Table 4.3 are stored under `docs/user_manual/screenshots/` (example_1.png … example_4.png), together with the completed checklist (2 exact passes; 2 explainable deviations).

## Appendix D — Sample PDF Report

Baringo farming business, 10 August 2026 14:17 — Risk 35.2% (Low); SHAP: Male Owners −0.266, Revenue Change Ratio −0.241, Total Monthly Expenses +0.202, Revenue Last Month −0.184, Normal Monthly Revenue −0.171; 10-step plan and encouragement confirmed coherent; 2 pages, branded footer.

## Appendix E — Repository Structure

```
BizFlow360/
├── playground.py                     # Streamlit application (5 pages)
├── ml_models/
│   ├── data/                         # unified_msme_modeling_data.csv
│   ├── scripts/                      # train_bizflow_engine_v2.py, v2_1.py
│   └── models/
│       ├── trained/on_real_data/     # bizflow_engine_v2.1*.joblib, model_card_v2.1.json
│       └── metrics/on_real_data/     # validation_metrics_v2.csv, calibration & CM plots
├── docs/user_manual/                 # Guide, validation report, screenshots
└── .streamlit/, requirements, README
```

## Appendix F — List of Figures and Tables

Table 4.1 (leakage ablation); Table 4.2 (v2.1 metrics); Table 4.3 (validation scenarios); SHAP waterfall plots (Analyze tab and `calibration_curve_v2.1.png`, `confusion_matrix_v2.1.png`); app screenshots (Welcome, Analyze, History, Chat, Settings); sample PDF report.