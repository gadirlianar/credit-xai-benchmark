# Data Governance and Provenance

This directory stores raw and preprocessed datasets for the research project:
*"Trade-Offs Between Real-Time Inference Latency and Explanation Faithfulness in Credit Risk Scoring"*

## Directory Structure
- `data/raw/`: Raw, unaltered source datasets. (Ignored by version control per privacy and redistribution governance).
- `data/processed/`: Deterministically cleaned, imputed, scaled, and partitioned datasets (`train.parquet`, `val.parquet`, `test.parquet`).

## Primary Dataset: Give Me Some Credit (GMSC)
- **Source**: Kaggle Competition / OpenML public benchmark (Dataset ID: 44080 / GMSC).
- **Domain**: Retail banking and consumer credit risk assessment.
- **Task**: Binary classification predicting serious financial default or 90+ days delinquency over a 2-year horizon (`SeriousDlqin2yrs` $\in \{0, 1\}$).
- **Number of Features**: 10 financial and demographic indicators (e.g., debt ratio, monthly income, revolving credit line utilization, past delinquency counts).
- **Target Distribution**: Approximately 6.68% default rate (realistic class imbalance).
- **Acquisition**: Can be downloaded automatically via `scripts/download_data.py`.
- **License**: Public for academic and educational research.
