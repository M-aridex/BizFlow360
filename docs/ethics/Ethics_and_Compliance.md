# ETHICS AND COMPLIANCE FRAMEWORK — BIZFLOW360

**BizFlow360: A Machine Learning-Based Early Warning System for Predicting Financial Distress Among Kenyan MSMEs**
**Team Maridex · Open University of Kenya · BSc Data Science · August 2026**
**Companion file: `Ethics_and_Compliance.md` (Version 1.0)**
**Document owner:** Marion Muthoni Mwendwa — Research, Documentation & Ethics Lead

---

## 1. Purpose and Scope

This document records the ethical principles, legal compliance measures, and responsible-AI controls applied across the BizFlow360 project — data sourcing, model development, explainability, generative-AI advisory, deployment, and user protection. It operationalises the commitments made in the project proposal (Section 3.14) and provides evidence of compliance for supervisors, examiners, and future maintainers.

**Scope:** KNBS MSME survey data used for training; the BizFlow360 Streamlit application; the NVIDIA NIM advisory layer; PDF reporting; and all project code and documentation hosted on GitHub.

## 2. Ethical Principles Adopted

The project aligns with the UNESCO Recommendation on the Ethics of Artificial Intelligence (2021), the AU Continental AI Strategy, and the Kenya National AI Strategy, operationalised through six principles:

1. **Fairness and non-discrimination** — no feature or threshold may intentionally disadvantage groups by gender, location, sector, or business size; disparities are monitored and documented.
2. **Transparency and explainability** — every prediction is accompanied by a SHAP explanation; model metrics, limitations, and design decisions are published in the model card and user guide.
3. **Privacy and data protection** — no personally identifiable information (PII) is collected, stored, or processed.
4. **Accountability** — version-controlled code, named team roles, documented incidents and remediations.
5. **Safety and reliability** — calibrated probabilities, validated operating thresholds, leakage audits, and graceful failure modes.
6. **Human oversight** — the system is decision-support only; final decisions remain with the business owner, lender, or advisor.

## 3. Data Protection and Privacy

### 3.1 Training Data
- Source: KNBS MSME survey microdata — **anonymized, publicly released, business-level** records.
- **No PII** (names, IDs, phone numbers, addresses) was accessed, downloaded, or used at any stage.
- Because no human-subjects primary data was collected, the project required no invasive procedures; the proposal's commitment to obtain consent *if* primary data is ever collected remains binding for future work.

### 3.2 Kenya Data Protection Act, 2019 — Compliance Mapping (Section 25 principles)

| DPA 2019 Principle | How BizFlow360 Complies |
|---|---|
| Lawfulness, fairness, transparency | Public anonymized data; clear in-app disclaimers; published methodology and validation report |
| Purpose limitation | Data used solely for distress prediction and decision support; no secondary use, no sale, no sharing |
| Data minimization | Only business-level financial indicators; app collects no PII; optional attachments used transiently and not stored server-side |
| Accuracy | Cleaning, imputation fitted on training folds only, outlier clipping, calibration, bootstrap confidence intervals |
| Storage limitation | User history stored **locally on the user's device only**; user may erase at any time; no persistent server-side user data |
| Security & confidentiality | API keys via Streamlit Secrets / environment variables; key rotation after a development exposure; `.gitignore` hygiene; no secrets in deployed code |
| Accountability | GitHub version control, model card, validation documentation, named Ethics Lead, incident log (Section 8) |

### 3.3 Data Subject Rights (as implemented in the app)

| Right | Implementation |
|---|---|
| Access | All stored data visible in-app (History page, Chat history) |
| Rectification | Users re-run analyses with corrected inputs at any time |
| Erasure | "Clear All History (Analysis & Chat)" removes local JSON stores |
| Portability | Branded PDF export of any report |

### 3.4 Secrets and Key Management
- The NVIDIA API key is developer-managed; end users never see or enter it.
- Keys are supplied via `.streamlit/secrets.toml` or environment variables; the file is git-ignored.
- A pre-deployment `grep` audit confirmed no key strings in the pushed repository.

## 4. Model Ethics and Fairness

### 4.1 Scientific Integrity (Target-Leakage Audit)
The v1.0 model inadvertently included profitability features that overlap definitionally with the distress label. This was detected, removed in v2.x, and quantified via ablation (ΔAUC ≈ 0.012). Publishing this audit — rather than hiding it — is a core integrity commitment.

### 4.2 Honest Probabilities (Calibration)
Scores are Platt-calibrated so a displayed "70%" is a truthful probability estimate, not an inflated ranking. Overstating certainty to users would be an ethical failure; calibration prevents it. Brier score 0.206 with 95% CIs is published.

### 4.3 Threshold Ethics (Recall Priority)
Financial distress prediction is sensitive to **false negatives** (a distressed business told it is safe). The operating threshold (τ = 0.33, max-F1; τ = 0.375 for recall ≥ 60%) deliberately favours recall (74.6%), accepting more false positives — the safer error direction for vulnerable MSMEs, since a false positive costs only "extra advice."

### 4.4 Fairness Considerations
- **County and sector** enter the model as descriptive baselines of structural economic differences (markets, climate, infrastructure), not as punitive proxies. Their influence is fully visible via SHAP.
- **Gender variables** (male/female working owners) are included to reflect ownership structure per the KNBS instrument; SHAP monitoring shows no systematic punitive effect against female-owned businesses in validation scenarios.
- **Known risk:** baseline features can encode historical disparities. **Mitigation:** transparency (SHAP), documentation, and a committed future disparate-impact audit (Section 10).

## 5. Explainability and Transparency

1. **Per-prediction SHAP waterfall** with plain-language meaning cards for every top driver.
2. **Model card** (in-app Settings + repository): model family, metrics with confidence intervals, operating threshold, leakage policy, limitations.
3. **User guide and validation report** (`docs/user_manual/`) with four documented scenarios, including two *explainable deviations* reported honestly rather than hidden.
4. **"Why this score?"** tab ensures no user ever receives a bare verdict.

## 6. Generative AI Governance (NVIDIA NIM Advisor)

The advisory layer (`z-ai/glm-5.2` text; `meta/llama-3.2-90b-vision-instruct` vision) operates under enforced system-prompt rules:

1. **Domain lock** — refuses non-financial topics politely and redirects.
2. **Evidence grounding** — must use only the numbers provided in the user context; never invent external facts about counties, markets, or competition.
3. **No promised outcomes or timeframes** — prohibits claims such as "your numbers will change in 2–3 months."
4. **Prediction–advice separation** — every output states the score is a statistical prediction and suggestions are not guarantees.
5. **No professional advice** — tax/legal matters are referred to certified professionals.
6. **Hallucination mitigation** — prompts are constrained to computed figures (deficit, break-even, SHAP drivers); outputs are streamed for user review and are editable/ignorable by the user.
7. **Vision consent** — images/receipts are processed only when the user actively uploads or captures them; they are used transiently for the response and not stored server-side.
8. **Reproducibility** — fixed seed (42) and documented temperature (0.7) for advisory calls.

## 7. Responsible Use and User Protection

- **Disclaimers** appear on the Welcome page, Analyze page, Chat, and every PDF footer: decision-support only; not financial, legal, or tax advice.
- **Accessible language** — friendly sector labels ("Duka, Mama Mboga, Kibanda"), KES examples, simple explanations; mobile-responsive UI for low-end devices.
- **No dark patterns** — no coercion, no data harvesting, no mandatory accounts.
- **Graceful degradation** — offline AI service shows a clear notice rather than silent failure or fabricated content.

## 8. Incident and Remediation Log

| # | Incident | When | Remediation | Status |
|---|---|---|---|---|
| 1 | API key present in development code | Aug 2026 | Key rotated; migrated to Secrets/env; repo audit; `.gitignore` updated | ✅ Closed |
| 2 | Target leakage in v1.0 feature set | Jul–Aug 2026 | Leakage-safe v2.x feature set; ablation published | ✅ Closed |
| 3 | PDF Unicode rendering bug (garbled text) | 10 Aug 2026 | Sanitization pipeline (emoji strip, punctuation map, latin-1 ignore); verified clean report | ✅ Closed |
| 4 | Low recall at default threshold (v2.0) | Aug 2026 | Operating-threshold selection; recall 74.6% at τ=0.33 | ✅ Closed |

## 9. Compliance Checklist (Pre-Defense)

| Control | Status |
|---|---|
| No PII in data or app | ✅ |
| DPA 2019 principles mapped | ✅ |
| Local-only user storage + erasure control | ✅ |
| Secrets management + rotation | ✅ |
| Leakage audit published | ✅ |
| Calibration + CIs published | ✅ |
| SHAP explanations for all predictions | ✅ |
| Generative-AI domain lock + evidence rules | ✅ |
| Disclaimers in UI and PDF | ✅ |
| Incident log maintained | ✅ |

## 10. Limitations and Ongoing Obligations

1. County/sector baselines require periodic **disparate-impact audits** as more data becomes available.
2. The advisory layer remains dependent on a third-party API; continuity planning (fallback models) is future work.
3. If primary data collection is ever undertaken, **informed consent** procedures and ODPC guidance will apply before collection.
4. Model drift monitoring (performance over time) is recommended post-deployment.

## 11. Governance and Sign-Off

- **Ethics Lead:** Marion Muthoni Mwendwa (owns this document and reviews changes).
- **ML Integrity:** Edusei Mikel Lisamba (leakage audits, calibration, model card).
- **Data Engineering:** Mutua Denis Mutio (data provenance, cleaning logs).
- **Deployment Security:** Yvette Akinyi Odeny (secrets, deployment, key rotation).

This document is version-controlled alongside the code; any material change to data use, model design, or advisory behaviour requires an update here before deployment.

---

# REFERENCES

Republic of Kenya. (2019). *The Data Protection Act, 2019.* Government of Kenya.

UNESCO. (2021). *Recommendation on the Ethics of Artificial Intelligence.* United Nations Educational, Scientific and Cultural Organization.

African Union Commission. (2024). *Continental Artificial Intelligence Strategy.* AUC.

Government of Kenya. (2025). *Kenya National Artificial Intelligence Strategy.* Ministry of Information, Communications and the Digital Economy.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model Cards for Model Reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)*, 220–229.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.

Kenya National Bureau of Statistics. (2016). *Kenya Micro, Small and Medium Enterprises Basic Report 2016.* KNBS.