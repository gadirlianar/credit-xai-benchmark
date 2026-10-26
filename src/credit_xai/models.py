import os
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from credit_xai.data import FEATURE_NAMES
from credit_xai.utils import get_logger, save_json, ensure_dirs

logger = get_logger("credit_xai.models")

class PyTorchMLP(nn.Module):
    """Deep feedforward neural network for tabular default prediction."""
    def __init__(self, input_dim: int = 10, hidden_dims: list = [64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

class MLPCreditModel:
    """Sklearn-compatible wrapper for PyTorch MLP."""
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: list = [64, 32],
        dropout: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 30,
        patience: int = 5,
        random_state: int = 42,
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device(device)
        self.model: Optional[PyTorchMLP] = None
        self.training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "MLPCreditModel":
        torch.set_num_threads(1)
        torch.manual_seed(self.random_state)
        self.model = PyTorchMLP(self.input_dim, self.hidden_dims, self.dropout).to(self.device)
        
        # Class weighting for BCE loss
        pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / max(sum(y_train), 1)], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32)))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_x_t = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        val_y_t = torch.from_numpy(y_val.astype(np.float32)).to(self.device)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        logger.info(f"Training MLP for up to {self.epochs} epochs...")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= len(X_train)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_x_t)
                val_loss = criterion(val_logits, val_y_t).item()

            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Fit Platt Scaling calibrator on validation logits
        self.model.eval()
        with torch.no_grad():
            val_logits_np = self.model(val_x_t).cpu().numpy().reshape(-1, 1)
        from sklearn.linear_model import LogisticRegression
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        self.calibrator.fit(val_logits_np, y_val)
        logger.info("Fitted Platt scaling calibrator for MLPCreditModel.")
        return self

    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            logits = self.model(x_t).cpu().numpy().reshape(-1, 1)
            if calibrated and hasattr(self, "calibrator") and self.calibrator is not None:
                probs = self.calibrator.predict_proba(logits)[:, 1]
            else:
                raw_sig = 1.0 / (1.0 + np.exp(-logits.ravel()))
                probs = raw_sig
            return np.column_stack([1.0 - probs, probs])

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X, calibrated=False)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> xgb.XGBClassifier:
    cfg = config or {}
    model = xgb.XGBClassifier(
        n_estimators=cfg.get("n_estimators", 200),
        max_depth=cfg.get("max_depth", 5),
        learning_rate=cfg.get("learning_rate", 0.05),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        random_state=cfg.get("random_state", 42),
        eval_metric="auc",
        early_stopping_rounds=cfg.get("early_stopping_rounds", 15),
        n_jobs=-1
    )
    logger.info("Training XGBoost Classifier...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> lgb.LGBMClassifier:
    cfg = config or {}
    model = lgb.LGBMClassifier(
        n_estimators=cfg.get("n_estimators", 200),
        max_depth=cfg.get("max_depth", 5),
        num_leaves=cfg.get("num_leaves", 31),
        learning_rate=cfg.get("learning_rate", 0.05),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        random_state=cfg.get("random_state", 42),
        verbosity=cfg.get("verbosity", -1),
        n_jobs=-1
    )
    logger.info("Training LightGBM Classifier...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    return model

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> LogisticRegression:
    cfg = config or {}
    model = LogisticRegression(
        C=cfg.get("C", 1.0),
        max_iter=cfg.get("max_iter", 1000),
        random_state=cfg.get("random_state", 42),
        solver="lbfgs"
    )
    logger.info("Training Logistic Regression baseline...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculates comprehensive classification and calibration metrics."""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y, probs))
    pr_auc = float(average_precision_score(y, probs))
    f1 = float(f1_score(y, preds, zero_division=0))
    precision = float(precision_score(y, preds, zero_division=0))
    recall = float(recall_score(y, preds, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y, preds))
    brier = float(brier_score_loss(y, probs))
    cm = confusion_matrix(y, preds).tolist()

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "sample_count": len(y)
    }

def find_optimal_threshold(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Finds decision threshold maximizing F1 score on validation set."""
    probs = model.predict_proba(X_val)[:, 1]
    best_th = 0.5
    best_f1 = 0.0
    for th in np.linspace(0.1, 0.9, 81):
        preds = (probs >= th).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    return best_th
