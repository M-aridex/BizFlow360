# CAPSTONE PROJECT PROPOSAL DOCUMENT — BIZFLOW360

**BizFlow360: A Machine Learning-Based Early Warning System for Predicting Financial Distress Among Kenyan MSMEs**

**Team Name:** Maridex
**Tagline:** Financial clarity for every MSME.

**Prepared by Team Maridex:**

1. EDUSEI MIKEL LISAMBA — ST01/0149/2025 (Team Leader / ML Integration Lead)
2. MUTUA DENIS MUTIO — ST01/0046/2025 (Data Engineering & EDA Lead)
3. MARION MUTHONI MWENDWA — ST01/0144/2025 (Research, Documentation & Ethics Lead)
4. YVETTE AKINYI ODENY — ST01/33452/2025 (MVP, Deployment & Presentation Lead)

**Supervisor:** Dr. Irene Sitawa
**Institution:** Open University of Kenya
**Programme:** BSc Data Science
**Submission Date:** August, 2026

---

## TABLE OF CONTENTS

1. Chapter One: Introduction — Background • Problem Statement • Objectives • Research Questions • Scope • Limitations • Significance
2. Chapter Two: Literature Review — Theoretical Literature • Empirical Literature • Research Gap
3. Chapter Three: Research Methodology — Research Design • Data Sources • Collection • Cleaning • Storage • EDA • Feature Engineering • Models • Validation • MVP • Deployment • Ethics
4. References

---

# CHAPTER ONE: INTRODUCTION

## 1.1 Background of the Study

Micro, Small, and Medium Enterprises (MSMEs) are the backbone of Kenya's economy. They contribute significantly to employment creation, income generation, innovation, and poverty reduction. National MSME reports estimate that Kenya has millions of MSMEs operating across sectors such as retail, agriculture, transport, manufacturing, food services, construction, and digital commerce, employing a large share of the Kenyan workforce.

Despite their critical economic role, many Kenyan MSMEs face serious financial challenges. These include poor record-keeping, limited access to credit, unpredictable cash flows, high operating costs, inadequate financial planning, and the absence of early warning systems for identifying business distress. Most MSMEs operate informally and do not maintain audited financial statements, making it difficult for financial institutions to assess their creditworthiness using traditional methods.

In recent years, digital financial services have transformed how Kenyan businesses operate. Mobile money platforms such as M-Pesa, digital lending applications, mobile banking, and electronic payment systems have created new forms of business data. However, many MSMEs still lack affordable and accessible tools that can analyse this information and provide meaningful financial insights. Most existing financial prediction systems are designed for large corporations or formal businesses with complete financial records.

Machine learning provides an opportunity to solve this problem by identifying patterns in historical business and financial data. Predictive models can be trained to identify early signs of financial distress, such as declining cash flow, increasing liabilities, poor loan repayment behaviour, reduced sales, or excessive expenses. If deployed properly, such systems can help MSME owners, lenders, Saccos, microfinance institutions, and policymakers make better decisions.

This project proposes the development of BizFlow360, a machine learning-based early warning system designed to predict financial distress among Kenyan MSMEs. The system will analyse key financial and operational indicators and provide users with a risk score and actionable insights. The final output will be a simple web-based dashboard where users can enter business information and receive predictions about the financial health of the business.

## 1.2 Problem Statement

Many MSMEs in Kenya fail or experience financial difficulties due to poor financial planning, limited access to credit, and lack of early warning systems for financial distress. Traditional financial assessment methods often rely on audited accounts, collateral, and formal credit history, which many Kenyan MSMEs do not have. As a result, many small businesses are unable to access financing, while lenders face high risks due to poor risk assessment tools.

Existing studies and financial prediction models have mainly focused on large corporations and formal enterprises. Limited work exists on machine learning-based financial distress prediction specifically tailored for Kenyan MSMEs, especially those operating in the informal or semi-formal sectors. This creates a research and practical gap that this project seeks to address.

Therefore, there is a need to develop a predictive system that can identify early signs of financial distress among Kenyan MSMEs using key financial indicators, business performance data, and machine learning techniques.

## 1.3 Research Objectives

### General Objective

To develop BizFlow360, a machine learning-based early warning system for predicting financial distress among Micro, Small, and Medium Enterprises in Kenya.

### Specific Objectives

1. To identify the key financial and operational indicators that contribute to financial distress among Kenyan MSMEs.
2. To build predictive machine learning models for classifying MSMEs as financially distressed or financially stable.
3. To compare the performance of different classification models, including Logistic Regression, Random Forest, XGBoost, and LightGBM.
4. To evaluate the selected model using classification metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
5. To deploy a prototype web-based prediction dashboard where users can enter business financial information and receive a financial distress risk score.

## 1.4 Research Questions

1. What are the key financial and operational indicators that predict financial distress among Kenyan MSMEs?
2. Which machine learning classification model performs best in predicting MSME financial distress?
3. How does the performance of complex models such as XGBoost and LightGBM compare with a baseline model such as Logistic Regression?
4. How can the trained model be deployed as a simple and accessible prototype for MSME owners and financial service providers?

## 1.5 Project Scope

This project will focus on developing a binary classification machine learning system that predicts whether an MSME is likely to experience financial distress. The system will use business-related data such as revenue, expenses, assets, liabilities, loan repayment behaviour, business age, sector, location, and cash flow indicators where available.

The project will cover the following:

1. Data collection from publicly available Kenyan MSME and finance-related datasets.
2. Data cleaning and preprocessing.
3. Exploratory data analysis.
4. Feature engineering.
5. Model development using classification algorithms.
6. Model evaluation and comparison.
7. Development of a Streamlit-based prototype dashboard.
8. Deployment of the prototype on a free or low-cost hosting platform.

The system will not provide legally binding financial advice or replace professional accounting and audit services. Instead, it will serve as a decision-support tool that provides early warning signals and insights.

## 1.6 Limitations

1. **Limited Access to Private MSME Data** — Many Kenyan MSMEs operate informally and may not have complete financial records, which may limit the availability of high-quality firm-level data.
2. **Data Quality Issues** — The available datasets may contain missing values, inconsistent entries, duplicate records, and outliers that require careful cleaning.
3. **Class Imbalance** — Financial distress cases may be fewer than stable cases, which may affect model performance if not properly handled.
4. **Generalization Challenges** — The model may perform differently across business sectors, counties, and economic conditions.
5. **Computational Constraints** — The project will rely on available student resources such as Google Colab, which may limit the complexity of models that can be trained.
6. **Ethical and Privacy Constraints** — The project must avoid using personally identifiable information unless proper consent is obtained, in line with the Kenya Data Protection Act, 2019.

## 1.7 Significance of the Study

This project is significant to several groups:

- **MSME Owners** — The system will help business owners identify early warning signs of financial distress and take corrective action before the situation worsens.
- **Lenders and Financial Institutions** — Banks, Saccos, microfinance institutions, and digital lenders can use the system to improve credit risk assessment for MSMEs that lack formal financial statements.
- **Policymakers** — The project can provide insights into the factors that affect MSME financial health in Kenya, helping policymakers design better support programmes.
- **Data Science Students and Researchers** — The project contributes to the growing field of financial machine learning in Africa and demonstrates how predictive analytics can be applied to real business problems.
- **The Academic Community** — The study addresses a research gap by focusing on Kenyan MSMEs, an area that has received limited attention compared to large corporate financial distress prediction.

---

# CHAPTER TWO: LITERATURE REVIEW

## 2.1 Theoretical Literature

### 2.1.1 Financial Distress Theory

Financial distress refers to a situation where a business struggles to meet its financial obligations, such as paying suppliers, repaying loans, covering operating expenses, or maintaining positive cash flow. Financial distress theory suggests that businesses often show warning signs before failure. These signs may include declining revenue, increasing debt, poor liquidity, negative cash flow, and rising expenses.

One of the earliest and most famous financial distress models is the Altman Z-Score model, developed by Edward Altman in 1968. The model used financial ratios to predict corporate bankruptcy. Although the original model was designed for large manufacturing firms, its core idea remains relevant: financial ratios and business indicators can be used to predict distress. For Kenyan MSMEs, financial distress may not always appear in formal financial statements. Instead, signs may be reflected in reduced sales, irregular mobile money inflows, inability to restock, delayed rent payments, or dependence on short-term digital loans.

### 2.1.2 Credit Risk Theory

Credit risk theory focuses on the probability that a borrower will fail to repay a loan or meet financial obligations. In MSME lending, credit risk is often high because many small businesses lack collateral, audited accounts, or formal credit histories.

Traditional credit scoring models rely on variables such as income, collateral, credit history, and debt-to-income ratio. However, many Kenyan MSMEs operate informally, making these models insufficient. Machine learning models can improve credit risk assessment by identifying complex patterns in alternative data, including business transaction behaviour, sector performance, business age, repayment history, and cash flow trends.

### 2.1.3 Machine Learning Theory

Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data and make predictions. In supervised learning, a model is trained using historical data where the outcome is already known. For this project, the outcome will be whether an MSME is financially distressed or not.

The project will focus on classification algorithms, which are used to predict categorical outcomes. The models to be used include:

1. **Logistic Regression** — A simple and interpretable baseline model used for binary classification.
2. **Random Forest** — An ensemble model that builds multiple decision trees and combines their predictions.
3. **XGBoost** — A powerful gradient boosting algorithm known for high performance in structured data tasks.
4. **LightGBM** — A fast and efficient gradient boosting framework suitable for large datasets.

These models are suitable for financial prediction because they can handle numerical and categorical variables, identify non-linear relationships, and provide measurable performance results.

## 2.2 Empirical Literature

Empirical studies in financial distress prediction have shown that machine learning models often outperform traditional statistical methods, especially when dealing with complex and imbalanced datasets.

Altman's Z-Score model demonstrated that financial ratios could be used to predict bankruptcy among large corporations. Later studies expanded this idea by using neural networks, decision trees, random forests, and gradient boosting models.

In credit risk prediction, research has shown that ensemble models such as Random Forest and XGBoost often perform better than Logistic Regression because they can capture complex interactions between variables. For example, a business may have high revenue but also high liabilities, and a simple rule-based system may fail to interpret this combination correctly. A machine learning model can learn such patterns from historical data.

In the Kenyan context, studies and reports on financial inclusion have shown that many MSMEs rely on mobile money, informal savings groups, digital credit, and personal funds. FinAccess surveys conducted in Kenya have provided important insights into household and business financial behavior. These surveys show that access to finance remains a challenge for many small businesses, especially those in rural and informal sectors.

However, most existing predictive tools are not designed specifically for Kenyan MSMEs. Many are built for formal corporations, listed companies, or businesses with complete accounting records. This creates a gap for a localized, MSME-focused early warning system.

### Summary of Empirical Studies

| Study / Source | Method / Focus | Key Finding |
|---|---|---|
| Altman (1968) | Discriminant analysis, US firms | Financial ratios predict bankruptcy |
| Breiman (2001) | Random Forests | Ensembles improve accuracy and stability |
| Chen and Guestrin (2016) | XGBoost, credit tasks | Gradient boosting excels on tabular data |
| Ke et al. (2017) | LightGBM | Fast training on large datasets |
| FinAccess / CBK and FSD Kenya | Financial inclusion surveys | MSMEs rely on mobile money; credit gaps persist |

## 2.3 Research Gap

Existing research on financial distress prediction has mainly focused on large corporations, listed companies, and formal enterprises with audited financial statements. Limited work exists on machine learning-based early warning systems for Kenyan MSMEs, particularly those in the informal and semi-formal sectors.

Additionally, many existing models are not accessible to ordinary business owners. They often remain as academic notebooks or technical scripts rather than user-friendly systems. This project addresses this gap by developing not only a predictive model but also a simple prototype dashboard that can be used by MSME owners and financial service providers.

Therefore, the research gap is:

> Limited machine learning-based financial distress prediction tools have been developed specifically for Kenyan MSMEs, and few systems provide an accessible early warning prototype for real-time use.

This gap provides the justification for the development of BizFlow360 by Team Maridex.

---

# CHAPTER THREE: RESEARCH METHODOLOGY

## 3.1 Research Design

This project will adopt a predictive analysis research design. The design is suitable because the main goal is to use historical or business-related data to predict a future outcome, which is whether an MSME is likely to experience financial distress.

The project will use a supervised machine learning approach. The dataset will contain input features such as revenue, expenses, assets, liabilities, business age, sector, location, loan repayment behaviour, and other relevant financial indicators. The target variable will be a binary label:

- 0 = Financially Stable
- 1 = Financially Distressed

The system will learn patterns from the data and predict the probability of financial distress for new or unseen MSME records.

### Conceptual Framework

The BizFlow360 pipeline follows this flow:

Data Sources (KNBS • CBK • FinAccess • World Bank • Kaggle) → Data Cleaning & EDA (missing values • outliers • RFM • correlations) → Feature Engineering (ratios • KPIs • encoded variables) → Model Training (LR • Random Forest • XGBoost • LightGBM) → Validation (metrics vs baseline • ROC-AUC • F1) → Deployment (Streamlit MVP • Cloud hosting)

## 3.2 Data Sources

The project will use publicly available and ethically sourced datasets related to Kenyan MSMEs, finance, and business performance. Possible data sources include:

| Source | Description | Access |
|---|---|---|
| KNBS | MSME surveys, economic surveys | KNBS Open Data Portal |
| Central Bank of Kenya | Financial inclusion and credit indicators | CBK publications |
| FinAccess Surveys | Mobile money usage, credit access, savings | CBK / FSD Kenya microdata |
| World Bank Enterprise Surveys | Firm performance, access to finance | microdata.worldbank.org |
| Kaggle | Credit risk / loan default datasets for prototyping | kaggle.com |
| Saccos / MFIs (where possible) | Loan repayment records | Formal data requests |

Where real-world data is limited, synthetic or publicly available proxy datasets will be used for prototyping and clearly disclosed in the final report.

## 3.3 Data Collection Procedure

Data will be collected through the following steps:

1. Identify relevant datasets from KNBS, CBK, FinAccess, World Bank, and Kaggle.
2. Download and organize the datasets into a project folder.
3. Review data dictionaries and variable descriptions.
4. Select features relevant to MSME financial health.
5. Merge datasets where necessary using common identifiers such as sector, county, business size, or time period.
6. Create a final dataset for modelling.

The final dataset will include features such as:

- Business sector
- County or location
- Business age
- Monthly revenue
- Monthly expenses
- Profit margin
- Assets
- Liabilities
- Loan amount
- Loan repayment status
- Cash flow indicators
- Mobile money transaction indicators, where available
- Number of employees
- Financial distress label

## 3.4 Data Cleaning Procedure

The collected data will be cleaned to ensure quality and reliability. The cleaning process will include:

1. **Handling Missing Values** — Numerical columns will be imputed using median or mean values. Categorical columns will be imputed using mode or a separate "Unknown" category.
2. **Removing Duplicates** — Duplicate records will be identified and removed.
3. **Handling Outliers** — Outliers will be detected using box plots and the Interquartile Range method. Extreme values will be capped or removed where appropriate.
4. **Correcting Inconsistent Data** — Inconsistent labels, misspellings, and incorrect formats will be corrected. Example: "Nairobi", "Nairobi County", and "NBO" will be standardized.
5. **Class Imbalance Treatment** — If the target variable is imbalanced, techniques such as class weighting, SMOTE, or undersampling may be used.
6. **Data Type Conversion** — Variables will be converted to appropriate data types. Example: currency values will be converted to numeric format.

## 3.5 Data Storage

The project data will be stored in an organized and secure manner. The following storage approach will be used:

- **Raw Data Folder** — Contains original downloaded datasets.
- **Processed Data Folder** — Contains cleaned and transformed datasets.
- **Model Folder** — Contains trained model files saved using .pkl or .joblib format.
- **GitHub Repository** — All code, notebooks, and documentation will be version-controlled using GitHub.
- **Cloud Storage** — Backup copies will be stored on Google Drive to prevent data loss.

Data will be stored in formats such as CSV, Parquet, or Excel depending on size and usability.

## 3.6 Exploratory Data Analysis

Exploratory Data Analysis (EDA) will be performed to understand the patterns, relationships, and distributions in the dataset.

The EDA process will include:

- **Distribution Analysis** — Histograms and density plots will be used to examine the distribution of variables such as revenue, expenses, profit, loan amount, and liabilities.
- **Correlation Analysis** — A correlation heatmap will be used to identify relationships between numerical variables. This will help detect multicollinearity and select important features.
- **Trend Analysis** — Time series plots may be used where time-based data is available to observe trends in revenue, expenses, loan defaults, or business performance.
- **RFM Analysis** — RFM analysis will be used to segment businesses based on financial behavior where transaction data is available: Recency (how recent the last transaction or repayment was), Frequency (how often the business makes transactions), and Monetary (the total value of transactions). This can help identify stable, active, at-risk, or inactive businesses.
- **Visualizations** — Histograms, box plots, heatmaps, bar charts, time series plots, and count plots for the target variable.

## 3.7 Feature Engineering

Feature engineering will be used to create meaningful variables that improve model performance. The following features may be created:

| Feature | Formula / Description |
|---|---|
| Profit Margin | Revenue − Expenses |
| Debt-to-Income Ratio | Total Liabilities ÷ Total Revenue |
| Expense Ratio | Total Expenses ÷ Total Revenue |
| Business Age Category | < 1 year • 1–3 years • > 3 years |
| Loan Risk Indicator | Based on repayment behaviour or outstanding debt |
| Cash Flow Indicator | Inflows minus outflows where transaction data is available |
| Encoded Variables | Label / one-hot encoding of sector, county, business size |

## 3.8 Model Development

The project will use classification models to predict whether an MSME is financially distressed. The models selected for comparison are:

1. **Logistic Regression** — This will serve as the baseline model. It is simple, interpretable, and useful for comparison.
2. **Random Forest** — A tree-based ensemble model that handles non-linear relationships.
3. **XGBoost** — A powerful gradient boosting model suitable for structured data.
4. **LightGBM** — A fast gradient boosting model that performs well on large datasets.

The models will be trained using a training set and evaluated using a testing set. The dataset will be split into 80% training data and 20% testing data. Stratified sampling will be used to ensure that the proportion of distressed and stable businesses remains balanced between the training and testing sets.

## 3.9 Model Evaluation Metrics

The models will be evaluated using the following classification metrics:

- **Accuracy** — The proportion of correct predictions.
- **Precision** — The proportion of predicted distressed MSMEs that are actually distressed.
- **Recall** — The ability of the model to identify all distressed MSMEs.
- **F1-Score** — The harmonic mean of precision and recall.
- **ROC-AUC Score** — Measures how well the model separates distressed and stable MSMEs.

Since financial distress prediction is sensitive to false negatives, recall will be given special attention. A false negative occurs when the model predicts that a business is stable, but the business is actually distressed. This can be dangerous because the business may fail to receive early warning support.

## 3.10 Baseline Model Comparison

A baseline model is important to determine whether the complex model is actually worth using. Logistic Regression will be used as the baseline model.

The expected comparison format will be:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | To be recorded | To be recorded | To be recorded | To be recorded | To be recorded |
| Random Forest | To be recorded | To be recorded | To be recorded | To be recorded | To be recorded |
| XGBoost | To be recorded | To be recorded | To be recorded | To be recorded | To be recorded |
| LightGBM | To be recorded | To be recorded | To be recorded | To be recorded | To be recorded |

## 3.11 Model Interpretability

To make the system more useful and transparent, the project will include model interpretation techniques such as:

- **Feature Importance** — Shows which variables contribute most to the prediction.
- **SHAP Values, where possible** — Helps explain why a specific MSME was classified as high risk or low risk.

This will help the system provide insights such as:

> "The business is considered high risk because expenses are too high relative to revenue and liabilities are increasing."

## 3.12 Development of the Minimum Viable Product (MVP)

The final deliverable will not be only a dataset and Jupyter Notebook. Team Maridex will develop a simple web-based prototype using Streamlit.

The BizFlow360 dashboard will allow users to enter business information such as:

- Monthly revenue
- Monthly expenses
- Total assets
- Total liabilities
- Loan amount
- Loan repayment status
- Business age
- Business sector
- County
- Number of employees

The system will then output:

1. **Financial Distress Risk Score** — Example: Risk of financial distress = 82%
2. **Risk Category** — Low Risk, Medium Risk, or High Risk
3. **Key Risk Factors** — Example: High expenses, low profit margin, and high liabilities are contributing to the risk.
4. **Recommended Actions** — Example: Reduce operating expenses, restructure debt, and improve cash flow monitoring.

This will make the project practical, user-friendly, and relevant to real business needs.

## 3.13 System Deployment

The prototype will be deployed using a free or low-cost platform to ensure accessibility. Possible deployment platforms include:

- Streamlit Community Cloud
- Hugging Face Spaces
- Render
- Gradio
- GitHub Pages, where suitable

The deployed system will allow lecturers, students, and potential users to interact with the model without installing Python or running code locally.

## 3.14 Ethical Considerations

The project will follow ethical data science practices.

1. **Data Privacy** — Personally identifiable information will not be used unless properly authorized.
2. **Kenya Data Protection Act, 2019** — The project will comply with the principles of the Kenya Data Protection Act, including lawful processing, data minimization, and purpose limitation.
3. **Informed Consent** — If primary data is collected, consent will be obtained from participants.
4. **Bias and Fairness** — The team will check whether the model unfairly disadvantages certain groups based on location, gender, sector, or business size.
5. **Transparency** — The system will provide explanations for predictions where possible.
6. **Responsible Use** — The system will be presented as a decision-support tool, not as a replacement for professional financial advice.

## 3.15 Expected Project Outputs

By the end of this project, Team Maridex expects to deliver the following:

1. A cleaned and structured MSME dataset.
2. An exploratory data analysis report with visualizations.
3. A feature-engineered dataset for modelling.
4. Multiple trained classification models.
5. A comparison of model performance against a baseline model.
6. A final selected model for predicting financial distress.
7. A Streamlit-based prototype dashboard named BizFlow360.
8. A final capstone report containing implementation, results, testing, conclusions, and recommendations.

---

# REFERENCES

Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance, 23*(4), 589–609.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Central Bank of Kenya. (n.d.). *Financial inclusion and FinAccess survey reports.* Central Bank of Kenya.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*.

Kenya National Bureau of Statistics. (2016). *Kenya Micro, Small and Medium Enterprises Basic Report 2016.* Kenya National Bureau of Statistics.

Republic of Kenya. (2019). *The Data Protection Act, 2019.* Government of Kenya.

World Bank. (n.d.). *Enterprise Surveys: Kenya data.* World Bank Group.