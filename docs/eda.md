# Exploratory Data Analysis (EDA) - Home Credit Default Risk

This document details the exploratory analysis of the primary dataset `application_train.csv`.

---

## 1. Dataset Shape and Types
* **Total Rows**: 307,511
* **Total Columns**: 122
* **Data Types**:
  * `float64`: 65 columns
  * `int64`: 41 columns
  * `object/str` (categorical): 16 columns

---

## 2. Target Variable and Class Imbalance
* **Target Variable**: `TARGET` (1 = Client with payment difficulties / default, 0 = All other cases)
* **Class Distribution**:
  * **Class 0 (Non-default)**: 282,686 (91.93%)
  * **Class 1 (Default)**: 24,825 (8.07%)
* **Imbalance Ratio**: ~11.4 : 1
* **Lending Risk Context**: A default rate of ~8% means that class imbalance must be handled carefully. Using standard accuracy as the optimization metric would result in a model that trivially predicts 0 for everything. We will use **ROC-AUC** and **PR-AUC** as evaluation metrics, and implement class weights in the loss function during FT-Transformer training.

---

## 3. Missing Value Analysis
A high percentage of columns have missing values. Out of 122 columns, 67 columns contain missing data.
Here are the top columns with missing values:
* `COMMONAREA_AVG` / `COMMONAREA_MODE` / `COMMONAREA_MEDI`: **69.87%**
* `NONLIVINGAPARTMENTS_AVG` / `NONLIVINGAPARTMENTS_MODE` / `NONLIVINGAPARTMENTS_MEDI`: **69.43%**
* `FONDKAPREMONT_MODE`: **68.39%**
* `LIVINGAPARTMENTS_AVG` / `LIVINGAPARTMENTS_MODE` / `LIVINGAPARTMENTS_MEDI`: **68.36%**
* `OWN_CAR_AGE`: **65.99%**

### External Sources (Key Predictive Features)
The `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` columns represent normalized scores from external data sources and are historically the most predictive features:
* `EXT_SOURCE_1`: **56.40% missing** (173,378 nulls)
* `EXT_SOURCE_2`: **0.21% missing** (660 nulls)
* `EXT_SOURCE_3`: **19.80% missing** (60,965 nulls)

We will engineer an `EXT_SOURCE_MEAN` feature to capture the average of the available scores, and use median imputation for the individual scores.

---

## 4. Anomalies and Data Cleaning
* **Days Birth (`DAYS_BIRTH`)**: Ranges from -25,229 to -7,489 (represented as negative days relative to application date). We will convert this to age in years: `AGE_YEARS = -DAYS_BIRTH / 365.25`.
* **Days Employed (`DAYS_EMPLOYED`)**:
  * Min: -17,912 days
  * Max: **365,243 days** (Anomaly!)
  * The value `365243` represents exactly 1,000 years, which is a placeholder representing "unemployed" or "not available". 
  * We will replace `365243` with `NaN` (which will be imputed with median or 0) and create a binary indicator column `DAYS_EMPLOYED_ANOM` to preserve this information.
  * For valid negative values, we will compute: `EMPLOYMENT_YEARS = -DAYS_EMPLOYED / 365.25`.

---

## 5. Planned Financial Features
To enhance the model's predictive power, we will engineer the following domain-specific ratios:
1. `AGE_YEARS`: Client's age.
2. `EMPLOYMENT_YEARS`: Client's employment duration.
3. `CREDIT_TO_INCOME`: Total credit amount relative to income (`AMT_CREDIT / AMT_INCOME_TOTAL`).
4. `ANNUITY_TO_INCOME`: Loan annuity relative to income (`AMT_ANNUITY / AMT_INCOME_TOTAL`).
5. `GOODS_TO_INCOME`: Price of goods relative to income (`AMT_GOODS_PRICE / AMT_INCOME_TOTAL`).
6. `INCOME_PER_FAMILY_MEMBER`: Family-size adjusted income (`AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`).
7. `EXT_SOURCE_MEAN`: Average of available external scores (`mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)`).
8. `MISSING_VALUE_COUNT`: Total count of missing values per application row.
