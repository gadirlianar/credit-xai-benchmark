#!/usr/bin/env python3
"""
Preprocesses data, trains XGBoost, LightGBM, MLP, and Logistic Regression models,
evaluates predictive performance on the held-out test partition, saves artifacts and plots Figure 2.
"""
import pickle
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from credit_xai.config import ExperimentConfig
from credit_xai.data import FEATURE_NAMES, TARGET_NAME
from credit_xai.preprocessing import partition_and_preprocess
from credit_xai.models import (
    train_xgboost,
    train_lightgbm,
    MLPCreditModel,
    train_logistic_regression,
    evaluate_model,
    find_optimal_threshold
)
from credit_xai.plotting import plot_predictive_performance
from credit_xai.utils import get_logger, save_json, ensure_dirs, set_seeds

logger = get_logger("train_models")

def main():
    cfg = ExperimentConfig.from_yaml("configs/experiment.yaml")
    set_seeds(cfg.project.seed)

    raw_path = Path(cfg.data.raw_dir) / cfg.data.raw_filename
    if not raw_path.exists():
        logger.error(f"Raw dataset not found at {raw_path}. Run download_data.py first.")
        return

    logger.info("Loading raw dataset...")
    df = pd.read_csv(raw_path)

    # 1. Partition and Preprocess
    train_df, val_df, test_df, preprocessor = partition_and_preprocess(
        df=df,
        output_dir=Path(cfg.data.processed_dir),
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.project.seed
    )

    X_train, y_train, _ = preprocessor.transform(train_df)
    X_val, y_val, _ = preprocessor.transform(val_df)
    X_test, y_test, _ = preprocessor.transform(test_df)

    # Model save directory
    model_dir = Path("results/metadata/models")
    ensure_dirs(model_dir)

    models = {}
    training_times = {}

    # 2. Train XGBoost
    t0 = time.perf_counter()
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, cfg.models.xgboost)
    training_times["XGBoost"] = round(time.perf_counter() - t0, 2)
    models["XGBoost"] = xgb_model
    with open(model_dir / "xgboost.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    # 3. Train LightGBM
    t0 = time.perf_counter()
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, cfg.models.lightgbm)
    training_times["LightGBM"] = round(time.perf_counter() - t0, 2)
    models["LightGBM"] = lgb_model
    with open(model_dir / "lightgbm.pkl", "wb") as f:
        pickle.dump(lgb_model, f)

    # 4. Train Multilayer Perceptron (MLP)
    t0 = time.perf_counter()
    mlp_model = MLPCreditModel(
        input_dim=len(FEATURE_NAMES),
        hidden_dims=cfg.models.mlp.get("hidden_dims", [64, 32]),
        dropout=cfg.models.mlp.get("dropout", 0.2),
        lr=cfg.models.mlp.get("learning_rate", 0.001),
        weight_decay=cfg.models.mlp.get("weight_decay", 1e-4),
        batch_size=cfg.models.mlp.get("batch_size", 128),
        epochs=cfg.models.mlp.get("epochs", 30),
        patience=cfg.models.mlp.get("early_stopping_patience", 5),
        random_state=cfg.project.seed,
        device=cfg.project.device
    )
    mlp_model.fit(X_train, y_train, X_val, y_val)
    training_times["MLP"] = round(time.perf_counter() - t0, 2)
    models["MLP"] = mlp_model
    with open(model_dir / "mlp.pkl", "wb") as f:
        pickle.dump(mlp_model, f)

    # 5. Train Logistic Regression
    t0 = time.perf_counter()
    lr_model = train_logistic_regression(X_train, y_train, cfg.models.logistic_regression)
    training_times["LogisticRegression"] = round(time.perf_counter() - t0, 2)
    models["LogisticRegression"] = lr_model
    with open(model_dir / "logistic_regression.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    # 6. Evaluate Models on Held-out Test Set
    logger.info("Evaluating models on held-out test partition...")
    results = {}
    roc_data = {}
    pr_data = {}

    for name, model in models.items():
        # Select threshold on validation set
        opt_thresh = find_optimal_threshold(model, X_val, y_val)
        metrics = evaluate_model(model, X_test, y_test, threshold=opt_thresh)
        metrics["optimal_threshold"] = round(opt_thresh, 3)
        metrics["training_time_sec"] = training_times[name]
        results[name] = metrics

        # Data for curves
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        prec, rec, _ = precision_recall_curve(y_test, probs)
        roc_data[name] = (fpr, tpr, metrics["roc_auc"])
        pr_data[name] = (rec, prec, metrics["pr_auc"])

        logger.info(
            f"[{name}] ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | Brier: {metrics['brier_score']:.4f} | "
            f"Train Time: {training_times[name]}s"
        )

    # Save Results Table 2
    table2_path = Path(cfg.paths["results_tables"]) / "table2_predictive_performance.json"
    ensure_dirs(table2_path.parent)
    save_json(results, table2_path)
    
    # Save Markdown and LaTeX versions of Table 2
    df_metrics = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    df_metrics.to_csv(Path(cfg.paths["results_tables"]) / "table2_predictive_performance.csv", index=False)
    
    # Plot Figure 2
    fig2_path = Path(cfg.paths["results_figures"]) / "figure2_predictive_performance.png"
    plot_predictive_performance(roc_data, pr_data, fig2_path)
    logger.info("Model training and evaluation pipeline completed successfully.")

if __name__ == "__main__":
    main()
