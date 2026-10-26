.PHONY: help env install data audit train smoke benchmark evaluate figures tables paper test all clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest

help:
	@echo "Credit Risk XAI Latency vs Faithfulness Benchmark"
	@echo "------------------------------------------------"
	@echo "make env         : Set up virtual environment with uv"
	@echo "make install     : Install dependencies"
	@echo "make data        : Download and prepare dataset"
	@echo "make audit       : Run dataset audit and validation"
	@echo "make train       : Train XGBoost, LightGBM, and MLP models"
	@echo "make smoke       : Run smoke test across all models and explainers"
	@echo "make benchmark   : Run latency and throughput benchmark"
	@echo "make evaluate    : Run faithfulness, stability, and Pareto evaluation"
	@echo "make figures     : Generate publication-ready figures"
	@echo "make tables      : Generate publication-ready summary tables"
	@echo "make test        : Run automated pytest test suite"
	@echo "make all         : Execute end-to-end pipeline"

install:
	/Users/anargadirli/.local/bin/uv pip install --python $(PYTHON) -r requirements.txt

data:
	$(PYTHON) scripts/download_data.py

audit:
	$(PYTHON) scripts/audit_dataset.py

train:
	$(PYTHON) scripts/train_models.py

smoke:
	$(PYTHON) scripts/run_smoke_tests.py

benchmark:
	$(PYTHON) scripts/run_benchmarks.py

evaluate:
	$(PYTHON) scripts/evaluate_explanations.py

pipeline:
	$(PYTHON) scripts/run_pipeline.py

test:
	$(PYTEST) tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
