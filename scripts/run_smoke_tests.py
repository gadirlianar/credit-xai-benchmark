#!/usr/bin/env python3
"""
Executes an end-to-end smoke test on a small sample to verify that all
models, explainers, metrics, and data structures execute without error.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import numpy as np
import pandas as pd

from credit_xai.config import ExperimentConfig
from credit_xai.data import FEATURE_NAMES, TARGET_NAME
from credit_xai.preprocessing import CreditDataPreprocessor
from credit_xai.explainers import (
    TreeSHAPExplainer,
    KernelSHAPExplainer,
    FastSHAPExplainer,
    LIMEExplainer
)
from credit_xai.faithfulness import FaithfulnessEvaluator
from credit_xai.robustness import RobustnessEvaluator
from credit_xai.utils import get_logger, set_seeds

logger = get_logger("smoke_tests")

def main():
    cfg = ExperimentConfig.from_yaml("configs/experiment.yaml")
    set_seeds(cfg.project.seed)
    logger.info("Starting smoke tests across models and explainers...")

    test_parquet = Path(cfg.data.processed_dir) / cfg.data.processed_test_filename
    train_parquet = Path(cfg.data.processed_dir) / cfg.data.processed_train_filename
    preprocessor_json = Path(cfg.data.processed_dir) / "preprocessor.json"
    model_dir = Path("results/metadata/models")

    if not test_parquet.exists() or not (model_dir / "xgboost.pkl").exists():
        logger.error("Preprocessed data or models missing. Run train_models.py first.")
        sys.exit(1)

    # Load data
    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)
    preprocessor = CreditDataPreprocessor()
    # Mock preprocessor state load or fit
    preprocessor.fit(train_df)

    X_train, y_train, _ = preprocessor.transform(train_df)
    X_test, y_test, _ = preprocessor.transform(test_df)

    # Load models
    with open(model_dir / "xgboost.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(model_dir / "lightgbm.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    with open(model_dir / "mlp.pkl", "rb") as f:
        mlp_model = pickle.load(f)

    # 5 test instances and 25 background samples for smoke test
    X_sample = X_test[:5]
    X_bg = X_train[:25]
    n_features = len(FEATURE_NAMES)

    smoke_results = {}

    # Test 1: TreeSHAP on XGBoost
    try:
        t_exp = TreeSHAPExplainer(xgb_model, background_data=X_bg)
        attr = t_exp.explain_instance(X_sample[0])
        assert attr.shape == (n_features,), f"Shape mismatch: {attr.shape}"
        assert not np.isnan(attr).any(), "NaN found in TreeSHAP attributions"
        smoke_results["TreeSHAP_XGBoost"] = "PASSED"
    except Exception as e:
        smoke_results["TreeSHAP_XGBoost"] = f"FAILED: {e}"

    # Test 2: TreeSHAP on LightGBM
    try:
        t_exp = TreeSHAPExplainer(lgb_model, background_data=X_bg)
        attr = t_exp.explain_instance(X_sample[0])
        assert attr.shape == (n_features,), f"Shape mismatch: {attr.shape}"
        smoke_results["TreeSHAP_LightGBM"] = "PASSED"
    except Exception as e:
        smoke_results["TreeSHAP_LightGBM"] = f"FAILED: {e}"

    # Test 3: KernelSHAP on MLP
    try:
        k_exp = KernelSHAPExplainer(mlp_model.predict_proba, background_data=X_bg, nsamples=20)
        attr = k_exp.explain_instance(X_sample[0])
        assert attr.shape == (n_features,), f"Shape mismatch: {attr.shape}"
        smoke_results["KernelSHAP_MLP"] = "PASSED"
    except Exception as e:
        smoke_results["KernelSHAP_MLP"] = f"FAILED: {e}"

    # Test 4: FastSHAP on XGBoost (train for 1 epoch for smoke test)
    try:
        f_exp = FastSHAPExplainer(xgb_model.predict_proba, background_data=X_bg, epochs=1, batch_size=16)
        f_exp.train_explainer(X_train[:100])
        attr = f_exp.explain_instance(X_sample[0])
        assert attr.shape == (n_features,), f"Shape mismatch: {attr.shape}"
        smoke_results["FastSHAP_XGBoost"] = "PASSED"
    except Exception as e:
        smoke_results["FastSHAP_XGBoost"] = f"FAILED: {e}"

    # Test 5: LIME on LightGBM
    try:
        l_exp = LIMEExplainer(lgb_model.predict_proba, training_data=X_train[:100], num_samples=50)
        attr, r2 = l_exp.explain_instance_with_score(X_sample[0])
        assert attr.shape == (n_features,), f"Shape mismatch: {attr.shape}"
        smoke_results["LIME_LightGBM"] = "PASSED"
    except Exception as e:
        smoke_results["LIME_LightGBM"] = f"FAILED: {e}"

    # Test 6: Faithfulness Evaluator
    try:
        evaluator = FaithfulnessEvaluator(xgb_model.predict_proba, background_data=X_bg)
        raw_curve, norm_curve, raw_auc, norm_auc = evaluator.compute_deletion_curve(X_sample[0], attr)
        assert len(raw_curve) == n_features + 1, f"Curve length mismatch: {len(raw_curve)}"
        assert 0.0 <= norm_auc <= 1.0, f"Normalized AUC out of bounds: {norm_auc}"
        smoke_results["FaithfulnessEvaluator"] = "PASSED"
    except Exception as e:
        smoke_results["FaithfulnessEvaluator"] = f"FAILED: {e}"

    # Test 7: Robustness Evaluator
    try:
        rob_evaluator = RobustnessEvaluator()
        sens = rob_evaluator.evaluate_input_perturbation_sensitivity(t_exp, X_sample[:2], epsilon=0.01)
        assert "mean_lipschitz_drift" in sens, "Sensitivity metric missing"
        smoke_results["RobustnessEvaluator"] = "PASSED"
    except Exception as e:
        smoke_results["RobustnessEvaluator"] = f"FAILED: {e}"

    logger.info("================ SMOKE TEST RESULTS ================")
    all_passed = True
    for test_name, status in smoke_results.items():
        logger.info(f"  - {test_name}: {status}")
        if "FAILED" in status:
            all_passed = False
    logger.info("====================================================")

    if not all_passed:
        logger.error("Some smoke tests failed!")
        sys.exit(1)
    else:
        logger.info("All smoke tests passed cleanly! System is ready for full benchmark execution.")

if __name__ == "__main__":
    main()
