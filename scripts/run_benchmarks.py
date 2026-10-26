#!/usr/bin/env python3
"""
Runs the comprehensive latency and throughput benchmark across
TreeSHAP, KernelSHAP, FastSHAP, and LIME for supported base models.
Records 1,000 observations per configuration, computes percentiles (p50, p95, p99),
evaluates SLAs (50ms, 100ms), and generates Figures 3 and 4.
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
import time
from dataclasses import asdict
import numpy as np
import pandas as pd

from credit_xai.config import ExperimentConfig
from credit_xai.data import FEATURE_NAMES
from credit_xai.preprocessing import CreditDataPreprocessor
from credit_xai.explainers import (
    TreeSHAPExplainer,
    KernelSHAPExplainer,
    FastSHAPExplainer,
    LIMEExplainer
)
from credit_xai.benchmark import LatencyBenchmarkHarness, BenchmarkSummary
from credit_xai.plotting import plot_latency_distributions, plot_tail_latency
from credit_xai.utils import get_logger, save_json, ensure_dirs, set_seeds

logger = get_logger("run_benchmarks")

def main():
    cfg = ExperimentConfig.from_yaml("configs/experiment.yaml")
    set_seeds(cfg.project.seed)

    test_parquet = Path(cfg.data.processed_dir) / cfg.data.processed_test_filename
    train_parquet = Path(cfg.data.processed_dir) / cfg.data.processed_train_filename
    model_dir = Path("results/metadata/models")

    if not test_parquet.exists():
        logger.error("Processed data missing. Run train_models.py first.")
        return

    logger.info("Loading preprocessed dataset and model artifacts...")
    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)

    preprocessor = CreditDataPreprocessor()
    preprocessor.fit(train_df)

    X_train, y_train, _ = preprocessor.transform(train_df)
    X_test, y_test, _ = preprocessor.transform(test_df)

    with open(model_dir / "xgboost.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(model_dir / "lightgbm.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    with open(model_dir / "mlp.pkl", "rb") as f:
        mlp_model = pickle.load(f)

    # Reference background distribution for shap/lime
    bg_size = cfg.explainers.background_size
    X_bg = X_train[:bg_size]

    harness = LatencyBenchmarkHarness(
        experiment_id="credit_risk_xai_benchmark",
        warmup_runs=cfg.benchmarking.warmup_runs,
        strict_sla_ms=cfg.benchmarking.strict_sla_ms,
        broad_sla_ms=cfg.benchmarking.broad_sla_ms
    )

    summaries = []
    timed_n = cfg.benchmarking.timed_samples

    # -------------------------------------------------------------
    # 1. TreeSHAP (XGBoost & LightGBM)
    # -------------------------------------------------------------
    logger.info("--- Initializing TreeSHAP explainers ---")
    tree_xgb = TreeSHAPExplainer(xgb_model, background_data=X_bg)
    s1 = harness.run_benchmark(
        model_name="XGBoost",
        explainer_name="TreeSHAP",
        predict_fn=xgb_model.predict_proba,
        explainer=tree_xgb,
        X_test=X_test,
        configuration="interventional",
        timed_samples=timed_n,
        batch_size=1
    )
    summaries.append(s1)

    tree_lgb = TreeSHAPExplainer(lgb_model, background_data=X_bg)
    s2 = harness.run_benchmark(
        model_name="LightGBM",
        explainer_name="TreeSHAP",
        predict_fn=lgb_model.predict_proba,
        explainer=tree_lgb,
        X_test=X_test,
        configuration="interventional",
        timed_samples=timed_n,
        batch_size=1
    )
    summaries.append(s2)

    # -------------------------------------------------------------
    # 2. FastSHAP (Trained Explainer Networks for XGBoost, LightGBM, MLP)
    # -------------------------------------------------------------
    logger.info("--- Training & Benchmarking FastSHAP explainers ---")
    # FastSHAP on XGBoost
    fast_xgb = FastSHAPExplainer(
        predict_fn=xgb_model.predict_proba,
        background_data=X_bg,
        hidden_dims=cfg.explainers.fastshap.get("hidden_dims", [128, 128]),
        epochs=cfg.explainers.fastshap.get("epochs", 25),
        batch_size=cfg.explainers.fastshap.get("batch_size", 64),
        device=cfg.project.device
    )
    fast_xgb.train_explainer(X_train[:2000])
    s3 = harness.run_benchmark(
        model_name="XGBoost",
        explainer_name="FastSHAP",
        predict_fn=xgb_model.predict_proba,
        explainer=fast_xgb,
        X_test=X_test,
        configuration="amortized_25ep",
        timed_samples=timed_n,
        batch_size=1
    )
    summaries.append(s3)

    # FastSHAP on LightGBM
    fast_lgb = FastSHAPExplainer(
        predict_fn=lgb_model.predict_proba,
        background_data=X_bg,
        epochs=cfg.explainers.fastshap.get("epochs", 25),
        batch_size=cfg.explainers.fastshap.get("batch_size", 64),
        device=cfg.project.device
    )
    fast_lgb.train_explainer(X_train[:2000])
    s4 = harness.run_benchmark(
        model_name="LightGBM",
        explainer_name="FastSHAP",
        predict_fn=lgb_model.predict_proba,
        explainer=fast_lgb,
        X_test=X_test,
        configuration="amortized_25ep",
        timed_samples=timed_n,
        batch_size=1
    )
    summaries.append(s4)

    # FastSHAP on MLP
    fast_mlp = FastSHAPExplainer(
        predict_fn=mlp_model.predict_proba,
        background_data=X_bg,
        epochs=cfg.explainers.fastshap.get("epochs", 25),
        batch_size=cfg.explainers.fastshap.get("batch_size", 64),
        device=cfg.project.device
    )
    fast_mlp.train_explainer(X_train[:2000])
    s5 = harness.run_benchmark(
        model_name="MLP",
        explainer_name="FastSHAP",
        predict_fn=mlp_model.predict_proba,
        explainer=fast_mlp,
        X_test=X_test,
        configuration="amortized_25ep",
        timed_samples=timed_n,
        batch_size=1
    )
    summaries.append(s5)

    # -------------------------------------------------------------
    # 3. KernelSHAP (Budget-controlled: 100 samples)
    # -------------------------------------------------------------
    logger.info("--- Initializing KernelSHAP explainers ---")
    # KernelSHAP is O(2^M) and takes ~100-200ms per instance; evaluate on 100 timed instances
    k_samples = min(100, timed_n)
    kernel_xgb = KernelSHAPExplainer(xgb_model.predict_proba, background_data=X_bg, nsamples=100)
    s6 = harness.run_benchmark(
        model_name="XGBoost",
        explainer_name="KernelSHAP",
        predict_fn=xgb_model.predict_proba,
        explainer=kernel_xgb,
        X_test=X_test,
        configuration="nsamples_100",
        timed_samples=k_samples,
        batch_size=1
    )
    summaries.append(s6)

    kernel_lgb = KernelSHAPExplainer(lgb_model.predict_proba, background_data=X_bg, nsamples=100)
    s_k_lgb = harness.run_benchmark(
        model_name="LightGBM",
        explainer_name="KernelSHAP",
        predict_fn=lgb_model.predict_proba,
        explainer=kernel_lgb,
        X_test=X_test,
        configuration="nsamples_100",
        timed_samples=k_samples,
        batch_size=1
    )
    summaries.append(s_k_lgb)

    kernel_mlp = KernelSHAPExplainer(mlp_model.predict_proba, background_data=X_bg, nsamples=100)
    s7 = harness.run_benchmark(
        model_name="MLP",
        explainer_name="KernelSHAP",
        predict_fn=mlp_model.predict_proba,
        explainer=kernel_mlp,
        X_test=X_test,
        configuration="nsamples_100",
        timed_samples=k_samples,
        batch_size=1
    )
    summaries.append(s7)

    # -------------------------------------------------------------
    # 4. LIME (Local surrogate: 200 samples)
    # -------------------------------------------------------------
    logger.info("--- Initializing LIME explainers ---")
    lime_samples = min(200, timed_n)
    
    lime_xgb = LIMEExplainer(xgb_model.predict_proba, training_data=X_train[:500], num_samples=200)
    s_l_xgb = harness.run_benchmark(
        model_name="XGBoost",
        explainer_name="LIME",
        predict_fn=xgb_model.predict_proba,
        explainer=lime_xgb,
        X_test=X_test,
        configuration="num_samples_200",
        timed_samples=lime_samples,
        batch_size=1
    )
    summaries.append(s_l_xgb)

    lime_lgb = LIMEExplainer(lgb_model.predict_proba, training_data=X_train[:500], num_samples=200)
    s8 = harness.run_benchmark(
        model_name="LightGBM",
        explainer_name="LIME",
        predict_fn=lgb_model.predict_proba,
        explainer=lime_lgb,
        X_test=X_test,
        configuration="num_samples_200",
        timed_samples=lime_samples,
        batch_size=1
    )
    summaries.append(s8)

    lime_mlp = LIMEExplainer(mlp_model.predict_proba, training_data=X_train[:500], num_samples=200)
    s9 = harness.run_benchmark(
        model_name="MLP",
        explainer_name="LIME",
        predict_fn=mlp_model.predict_proba,
        explainer=lime_mlp,
        X_test=X_test,
        configuration="num_samples_200",
        timed_samples=lime_samples,
        batch_size=1
    )
    summaries.append(s9)

    # -------------------------------------------------------------
    # Save Benchmark Outputs
    # -------------------------------------------------------------
    raw_obs_path = Path(cfg.paths["results_raw"]) / "timing_observations.csv"
    harness.save_raw_observations(raw_obs_path)

    summary_dicts = [asdict(s) for s in summaries]
    summary_df = pd.DataFrame(summary_dicts)

    table4_json = Path(cfg.paths["results_tables"]) / "table4_latency_statistics.json"
    table4_csv = Path(cfg.paths["results_tables"]) / "table4_latency_statistics.csv"
    ensure_dirs(table4_json.parent)
    save_json(summary_dicts, table4_json)
    summary_df.to_csv(table4_csv, index=False)
    logger.info(f"Saved Table 4 latency statistics to {table4_csv}")

    # Plot Figures 3 & 4
    obs_df = pd.DataFrame([asdict(o) for o in harness.observations])
    fig3_path = Path(cfg.paths["results_figures"]) / "figure3_latency_distributions.png"
    fig4_path = Path(cfg.paths["results_figures"]) / "figure4_tail_latency.png"
    plot_latency_distributions(obs_df, fig3_path)
    plot_tail_latency(summary_df, fig4_path)

    # Print summary table
    logger.info("================ LATENCY BENCHMARK SUMMARY (Table 4) ================")
    cols_to_show = ["model_name", "explainer_name", "xai_p50_ms", "xai_p95_ms", "xai_p99_ms", "throughput_qps", "strict_sla_50ms_pct"]
    print(summary_df[cols_to_show].to_string(index=False))
    logger.info("====================================================================")

if __name__ == "__main__":
    main()
