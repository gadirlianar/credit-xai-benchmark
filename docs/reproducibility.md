# Reproducibility Guide

This document details the exact protocol for replicating all experiments, benchmarks, statistical analyses, figures, and LaTeX tables in this repository.

## 1. System Requirements & Environment
- **Operating System**: macOS (tested on Apple Silicon arm64) or Linux (Ubuntu 22.04+).
- **Python Version**: Python 3.12 (managed via `uv` or standard virtual environment).
- **Disk Space**: ~2 GB for virtual environment, dataset, models, and benchmark artifacts.
- **RAM**: Minimum 8 GB recommended (16 GB used in primary benchmark).

## 2. Quickstart Execution
To run the complete pipeline from scratch using a single command:
```bash
make all
```

Or step-by-step:

### Step 1: Virtual Environment Setup
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Step 2: Data Acquisition & Audit
```bash
python scripts/download_data.py
python scripts/audit_dataset.py
```

### Step 3: Model Training & Evaluation
```bash
python scripts/train_models.py
```

### Step 4: Smoke Testing Models & Explainers
```bash
python scripts/run_smoke_tests.py
```

### Step 5: High-Precision Latency Benchmark (p50, p95, p99)
```bash
python scripts/run_benchmarks.py
```

### Step 6: Faithfulness & Robustness Evaluation
```bash
python scripts/evaluate_explanations.py
```

### Step 7: Run Unit Tests
```bash
pytest tests/ -v
```

## 3. Determinism & Random Seeds
All stochastic operations (NumPy, PyTorch, Scikit-learn, XGBoost, LightGBM, and explainer samplers) are seeded with `seed: 42` by default. Multi-seed robustness experiments explicitly iterate over documented seeds: `[42, 123, 456, 789, 1011]`.
