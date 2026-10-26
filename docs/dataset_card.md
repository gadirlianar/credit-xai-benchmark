# Dataset Card: Give Me Some Credit (GMSC) Benchmark

## 1. Dataset Summary
The Give Me Some Credit dataset is an industry-standard open benchmark for credit risk scoring and probability of default (PD) estimation. The dataset represents historical borrower profiles with financial metrics, past payment delinquencies, and existing debt lines.

## 2. Feature Dictionary
| Feature Name | Type | Description | Unit / Range |
|---|---|---|---|
| `SeriousDlqin2yrs` (Target) | Binary | Person experienced 90 days past due delinquency or worse in 2 years | {0, 1} |
| `RevolvingUtilizationOfUnsecuredLines` | Continuous | Total balance on credit cards and personal lines divided by sum of credit limits | Ratio (normally 0–1, outliers clipped) |
| `age` | Integer | Age of borrower in years | Years (18–100) |
| `NumberOfTime30-59DaysPastDueNotWorse` | Integer | Number of times borrower has been 30–59 days past due in past 2 years | Count |
| `DebtRatio` | Continuous | Monthly debt payments, alimony, living costs divided by monthly gross income | Ratio |
| `MonthlyIncome` | Continuous | Monthly income | USD |
| `NumberOfOpenCreditLinesAndLoans` | Integer | Number of open loans (installment and credit cards) | Count |
| `NumberOfTimes90DaysLate` | Integer | Number of times borrower has been 90 or more days past due | Count |
| `NumberRealEstateLoansOrLines` | Integer | Number of mortgage and real estate loans | Count |
| `NumberOfTime60-89DaysPastDueNotWorse` | Integer | Number of times borrower has been 60–89 days past due in past 2 years | Count |
| `NumberOfDependents` | Integer | Number of dependents in family excluding self | Count |

## 3. Data Governance & Preprocessing
- **Missing Value Handling**: `MonthlyIncome` (~19.8% missing) and `NumberOfDependents` (~2.6% missing) are imputed using median values estimated strictly on the training partition. A missingness indicator flag is preserved.
- **Outlier Mitigation**: Heavy financial outliers (e.g. debt ratio > 3000, revolving utilization > 13) are winsorized/clipped based on training set percentiles to prevent numerical instability.
- **Data Splitting**: Stratified 70/15/15 split preserving default class proportions across train, validation, and test subsets.
- **Leakage Prevention**: All transformations (imputation medians, robust scalers) are fitted strictly on the training set and applied out-of-sample to validation and test sets.
