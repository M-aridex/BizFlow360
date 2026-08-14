# CHAPTER TWO: LITERATURE REVIEW — EXPANDED

**BizFlow360: A Machine Learning-Based Early Warning System for Predicting Financial Distress Among Kenyan MSMEs**
**Team Maridex · Open University of Kenya · BSc Data Science · August 2026**
**Companion file: `literature.md`**

---

## 2.0 Overview

This chapter reviews the theoretical and empirical foundations of BizFlow360. It covers financial distress theory, credit risk and information asymmetry, machine learning (ML) theory for tabular financial data, and explainable artificial intelligence (XAI). It then synthesises empirical evidence on ML for bankruptcy and credit scoring, MSME finance in Kenya and developing economies, and the use of alternative digital data for credit assessment, before stating the research gap this project fills.

## 2.1 Theoretical Literature

### 2.1.1 Financial Distress Theory

Financial distress refers to a situation in which a business struggles to meet its financial obligations — paying suppliers, servicing loans, covering operating costs, or maintaining positive cash flow. Distress theory holds that businesses display observable warning signs before failure, including declining revenue, rising debt, poor liquidity, negative cash flow, and escalating expenses.

The foundational model is the **Altman Z-Score** (Altman, 1968), which used financial ratios and discriminant analysis to predict corporate bankruptcy. Its core insight remains valid: financial ratios and business indicators carry predictive power for distress. Later extensions (Altman & Hotchkiss, 2006) adapted the framework to modern bankruptcy prediction.

For Kenyan MSMEs, distress rarely appears in audited statements. Instead, warning signs manifest informally: reduced sales, irregular mobile-money inflows, inability to restock, delayed rent payments, shrinking inventory, and dependence on short-term digital loans. BizFlow360's feature design (revenue change ratio, expense-to-revenue ratio, stock change, closure history) operationalises this informal-distress view.

### 2.1.2 Credit Risk and Information Asymmetry

Credit risk theory examines the probability that a borrower fails to meet obligations. Merton's (1974) structural model formalised default as a function of firm value relative to liabilities. In MSME lending, however, the binding constraint is **information asymmetry**: Stiglitz and Weiss (1981) showed that when lenders cannot observe borrower risk, credit rationing results — borrowers are denied loans regardless of willingness to pay higher rates. This explains why many Kenyan MSMEs, lacking collateral, audited accounts, and formal credit histories, are excluded from finance even when viable.

Traditional credit scoring relies on income, collateral, and credit history (Thomas, 2000) — variables most informal MSMEs do not possess. Machine learning offers a way out by extracting risk signals from alternative and behavioural data, directly addressing the information asymmetry at the heart of MSME credit rationing.

### 2.1.3 Machine Learning Theory for Tabular Financial Data

In supervised learning, a model learns a mapping from labelled historical examples to predict outcomes for unseen records (Hastie, Tibshirani & Friedman, 2009). For binary distress classification, this project considers:

1. **Logistic Regression** — a linear, interpretable baseline (Thomas, 2000).
2. **Random Forest** — bagged decision trees that reduce variance and capture non-linearities (Breiman, 2001).
3. **Gradient Boosting** — stage-wise additive modelling that sequentially corrects errors (Friedman, 2001).
4. **XGBoost** — regularised, second-order gradient boosting with strong performance on structured data (Chen & Guestrin, 2016).
5. **LightGBM** — histogram-based, leaf-wise boosting that is fast and memory-efficient on large datasets (Ke et al., 2017).

Empirical benchmarking literature (Section 2.2) consistently finds tree ensembles superior on tabular credit data, motivating their selection here.

**Probability calibration.** Raw classifier scores are not always true probabilities. Platt scaling (Platt, 1999) fits a sigmoid to map scores to calibrated probabilities — essential for BizFlow360, where a displayed "70% risk" must be honest and actionable for a business owner.

### 2.1.4 Explainable Artificial Intelligence (XAI)

A risk score without an explanation is of limited use to an MSME owner and raises accountability concerns. **SHAP** (SHapley Additive exPlanations; Lundberg & Lee, 2017) assigns each feature a contribution grounded in cooperative game theory (Shapley, 1953), providing consistent, locally accurate attributions. Alternative post-hoc methods such as LIME (Ribeiro, Singh & Guestrin, 2016) approximate models locally but lack SHAP's theoretical guarantees. BizFlow360 adopts SHAP waterfall plots so every prediction is transparently decomposed into risk-increasing and risk-reducing factors.

## 2.2 Empirical Literature

### 2.2.1 Machine Learning in Bankruptcy and Credit Scoring

Empirical work shows ML models generally outperform traditional statistical methods on distress and credit tasks, particularly with complex, imbalanced data:

- **Altman (1968)** demonstrated ratio-based bankruptcy prediction for US manufacturers.
- **Baesens et al. (2003)** benchmarked eight classifiers on real credit data; neural networks, SVMs, and ensembles outperformed logistic regression.
- **Khandani, Kim & Lo (2010)** showed ML consumer credit-risk models could reduce bank losses substantially.
- **Lessmann et al. (2015)**, updating the benchmark, found boosting-family models (XGBoost/LightGBM-style) consistently top-ranked.
- **Thomas (2000)** documented the industrial adoption of scoring models, validating the decision-support paradigm.

### 2.2.2 MSME Finance and Inclusion in Kenya and Developing Economies

- **Beck, Demirguc-Kunt & Levine (2005)** found SME development associated with growth and poverty reduction, establishing the macroeconomic stakes.
- **KNBS (2016)** estimates millions of Kenyan MSMEs providing the majority of non-agricultural employment, yet facing finance, records, and planning constraints.
- **FinAccess surveys (CBK & FSD Kenya, 2021)** and the **Global Findex (Demirgüç-Kunt et al., 2018)** show mobile money has deepened access, but MSME credit gaps persist, especially for informal and rural firms.
- **World Bank Enterprise Surveys** consistently rank access to finance among the top constraints for Kenyan firms.

### 2.2.3 Alternative Data and ML for MSME Credit in Africa

Where formal records are absent, digital footprints substitute for collateral. **Björkegren & Grissen (2018)** showed mobile-phone transaction data can predict repayment among unbanked borrowers in Rwanda, proving alternative-data credit scoring viable in African MSME contexts. Kenya's M-Pesa ecosystem makes similar signals available, motivating features such as revenue regularity and revenue-change ratios in BizFlow360.

### 2.2.4 Summary of Empirical Studies

| Study / Source | Method / Focus | Key Finding |
|---|---|---|
| Altman (1968) | Discriminant analysis, US firms | Financial ratios predict bankruptcy |
| Merton (1974) | Structural credit model | Default driven by firm value vs liabilities |
| Stiglitz & Weiss (1981) | Theory of credit markets | Information asymmetry causes credit rationing |
| Thomas (2000) | Survey of scoring | Scoring models standard in consumer credit |
| Breiman (2001) | Random Forests | Ensembles improve accuracy and stability |
| Friedman (2001) | Gradient boosting | Stage-wise additive modelling |
| Baesens et al. (2003) | Classifier benchmark | NN/SVM/ensembles beat logistic regression |
| Chen & Guestrin (2016) | XGBoost | State-of-the-art on tabular credit tasks |
| Ke et al. (2017) | LightGBM | Fast, scalable, accurate boosting |
| Khandani et al. (2010) | ML consumer credit | Non-linear models cut lender losses |
| Lessmann et al. (2015) | Benchmark update | Boosting family dominates credit scoring |
| Platt (1999) | Sigmoid calibration | Scores mapped to true probabilities |
| Shapley (1953); Lundberg & Lee (2017) | Game-theoretic attribution | Consistent per-prediction explanations (SHAP) |
| Björkegren & Grissen (2018) | Digital credit scoring, Rwanda | Mobile data predicts repayment among unbanked |
| Beck et al. (2005) | Cross-country SME study | SME growth linked to poverty reduction |
| KNBS (2016) | National MSME survey | MSMEs central to Kenyan employment; finance gaps |
| FinAccess (2021) | Household/business survey | Mobile money deepens inclusion; credit gaps persist |

## 2.3 Research Gap

Existing distress-prediction research concentrates on large, listed, or formal corporations with audited statements. Few studies target **Kenyan MSMEs**, particularly informal and semi-formal firms; fewer still produce **accessible, deployed tools** rather than academic notebooks; and almost none combine **leakage-aware modelling, calibrated probabilities, threshold-validated recall, and SHAP explainability** with a **generative-AI advisory layer** in one system.

> **Research gap:** Limited machine learning–based financial distress prediction tools have been developed specifically for Kenyan MSMEs, and few systems provide an accessible, explainable early-warning prototype for real-time use.

BizFlow360 addresses this gap end to end.

## 2.4 Conceptual Framework

The literature maps onto the BizFlow360 pipeline as follows:

    Data Sources (KNBS • CBK • FinAccess • World Bank • Kaggle)
        → Data Cleaning & EDA (missing values • outliers • RFM • correlations)
        → Feature Engineering (ratios • KPIs • encoded variables)
        → Model Training (LR • Random Forest • XGBoost • LightGBM)
        → Validation (metrics vs baseline • ROC-AUC • F1 • calibration)
        → Explainability (SHAP) & Advisory (NVIDIA NIM)
        → Deployment (Streamlit MVP • Cloud hosting)

- **Distress theory** informs the feature set (revenue dynamics, expense burden, stock velocity, closure history).
- **Credit-risk / information-asymmetry theory** defines the prediction target and the inclusion problem the tool addresses.
- **ML theory** justifies the model family and baseline comparison (Objective 3).
- **Calibration theory** ensures scores are honest probabilities.
- **XAI theory** delivers per-prediction transparency (Section 3.11 of the proposal).
- **Decision-support framing** shapes the MVP as advice, not binding financial counsel (ethical compliance, Section 3.14).

---

# REFERENCES

Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance, 23*(4), 589–609.

Altman, E. I., & Hotchkiss, E. (2006). *Corporate Financial Distress and Bankruptcy* (3rd ed.). John Wiley & Sons.

Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. *Journal of the Operational Research Society, 54*(6), 627–635.

Beck, T., Demirguc-Kunt, A., & Levine, R. (2005). SMEs, growth, and poverty: Cross-country evidence. *Journal of Economic Growth, 10*(3), 199–229.

Björkegren, D., & Grissen, D. (2018). *Machine learning for credit scoring in emerging markets.* Working Paper.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Central Bank of Kenya & FSD Kenya. (2021). *FinAccess Household Survey 2021.*

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.

Demirgüç-Kunt, A., Klapper, L., Singer, D., Ansar, S., & Hess, J. (2018). *The Global Findex Database 2017: Measuring Financial Inclusion and the Fintech Revolution.* World Bank.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189–1232.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*.

Kenya National Bureau of Statistics. (2016). *Kenya Micro, Small and Medium Enterprises Basic Report 2016.* KNBS.

Khandani, A. E., Kim, A. J., & Lo, A. W. (2010). Consumer credit-risk models via machine-learning algorithms. *Journal of Banking & Finance, 34*(11), 2767–2787.

Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research, 247*(1), 124–136.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.

Merton, R. C. (1974). On the pricing of corporate debt: The risk structure of interest rates. *The Journal of Finance, 29*(2), 449–470.

Muthinja, M. N., & Chipeta, C. (2018). Factors affecting financial inclusion among SMEs in Kenya. *Journal of African Business, 19*(4).

Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*. MIT Press.

Republic of Kenya. (2019). *The Data Protection Act, 2019.* Government of Kenya.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD*, 1135–1144.

Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games, 2*, 307–317.

Stiglitz, J. E., & Weiss, A. (1981). Credit rationing in markets with imperfect information. *The American Economic Review, 71*(3), 393–410.

Thomas, L. C. (2000). A survey of credit and behavioural scoring: Forecasting financial risk involving lending to consumers. *International Journal of Forecasting, 16*(2), 145–172.

World Bank. (n.d.). *Enterprise Surveys: Kenya data.* World Bank Group.