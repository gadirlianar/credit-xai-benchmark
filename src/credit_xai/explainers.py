import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import shap
import lime
import lime.lime_tabular
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from credit_xai.data import FEATURE_NAMES
from credit_xai.utils import get_logger

logger = get_logger("credit_xai.explainers")

class BaseExplainer(ABC):
    """Abstract base interface for all credit risk explainers."""
    def __init__(self, model: Any, feature_names: List[str] = FEATURE_NAMES):
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)

    @abstractmethod
    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        """Computes feature attribution vector of shape (n_features,) for a single instance."""
        pass

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        """Computes feature attributions of shape (N, n_features). Default falls back to loop."""
        return np.array([self.explain_instance(X[i]) for i in range(len(X))])

class TreeSHAPExplainer(BaseExplainer):
    """
    Polynomial-time TreeSHAP explainer for tree ensembles (XGBoost, LightGBM).
    Uses interventional background data to eliminate feature dependency assumptions.
    """
    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: List[str] = FEATURE_NAMES,
        feature_perturbation: str = "interventional",
        model_output: str = "probability"
    ):
        super().__init__(model, feature_names)
        self.background_data = background_data
        logger.info(f"Initializing TreeSHAP with {len(background_data)} background samples (output={model_output})...")
        self.explainer = shap.TreeExplainer(
            self.model,
            data=self.background_data,
            feature_perturbation=feature_perturbation,
            model_output=model_output
        )

    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        x_2d = x.reshape(1, -1)
        values = self.explainer.shap_values(x_2d)
        return self._normalize_shap_output(values)[0]

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        values = self.explainer.shap_values(X)
        return self._normalize_shap_output(values)

    def _normalize_shap_output(self, values: Any) -> np.ndarray:
        """Handles format differences across SHAP versions (list, 3D array, or 2D array)."""
        if isinstance(values, list):
            # Binary classification: positive class attribution is index 1
            arr = np.array(values[1] if len(values) > 1 else values[0])
        elif isinstance(values, np.ndarray):
            if values.ndim == 3:
                arr = values[:, :, 1]
            else:
                arr = values
        else:
            arr = np.array(values)
        return arr

class ExactShapleyExplainer(BaseExplainer):
    """
    Exact Shapley value explainer computing all 2^d subset coalitions.
    Feasible and exact for low-dimensional tabular models (d <= 14).
    Evaluates interventional expectation with respect to an empirical background baseline.
    """
    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
        feature_names: List[str] = FEATURE_NAMES
    ):
        super().__init__(None, feature_names)
        self.predict_fn = predict_fn
        self.baseline = np.median(background_data, axis=0)
        self.d = self.n_features
        import math

        # Precompute all 2^d binary masks and feature transition pairs
        self.masks = np.array([[int(b) for b in format(i, f'0{self.d}b')] for i in range(1 << self.d)])
        self.pairs = {i: [] for i in range(self.d)}
        for idx, m in enumerate(self.masks):
            k = int(np.sum(m))
            for i in range(self.d):
                if m[i] == 0:
                    m_with = m.copy()
                    m_with[i] = 1
                    idx_with = int(''.join(map(str, m_with)), 2)
                    weight = math.factorial(k) * math.factorial(self.d - k - 1) / math.factorial(self.d)
                    self.pairs[i].append((idx, idx_with, weight))

    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        hybrids = np.where(self.masks == 1, x.reshape(1, -1), self.baseline.reshape(1, -1))
        preds = self.predict_fn(hybrids)
        if preds.ndim == 2:
            preds = preds[:, 1]
        elif preds.ndim == 1 and len(preds) == len(self.masks):
            pass
        else:
            preds = preds.ravel()

        attributions = np.zeros(self.d)
        for i in range(self.d):
            for idx_without, idx_with, w in self.pairs[i]:
                attributions[i] += w * (preds[idx_with] - preds[idx_without])

        return attributions

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.explain_instance(X[i]) for i in range(len(X))])

class KernelSHAPExplainer(BaseExplainer):
    """
    Model-agnostic KernelSHAP explainer approximating Shapley values
    via weighted local linear regression over coalition samples.
    """
    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
        feature_names: List[str] = FEATURE_NAMES,
        nsamples: int = 100
    ):
        super().__init__(None, feature_names)
        self.predict_fn = predict_fn
        self.background_data = background_data
        self.nsamples = nsamples
        
        # Link function for probability space
        self.explainer = shap.KernelExplainer(self._predict_positive_class, self.background_data)

    def _predict_positive_class(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict_fn(X)
        if preds.ndim == 2 and preds.shape[1] >= 2:
            return preds[:, 1]
        return preds.ravel()

    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        x_2d = x.reshape(1, -1)
        values = self.explainer.shap_values(x_2d, nsamples=self.nsamples, silent=True)
        if isinstance(values, list):
            arr = np.array(values[1] if len(values) > 1 else values[0])
        else:
            arr = np.array(values)
        return arr[0]

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        values = self.explainer.shap_values(X, nsamples=self.nsamples, silent=True)
        if isinstance(values, list):
            return np.array(values[1] if len(values) > 1 else values[0])
        return np.array(values)

class FastSHAPNetwork(nn.Module):
    """Neural network parameterized to output Shapley value estimates in a single forward pass."""
    def __init__(self, n_features: int = 10, hidden_dims: list = [128, 128]):
        super().__init__()
        layers = []
        prev_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FastSHAPExplainer(BaseExplainer):
    """
    Amortized Shapley value explainer (Jethani et al., ICLR 2022).
    Trains an explainer network phi(x) that outputs Shapley attributions in O(1) time.
    """
    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
        feature_names: List[str] = FEATURE_NAMES,
        hidden_dims: List[int] = [128, 128, 64],
        lr: float = 0.002,
        batch_size: int = 128,
        epochs: int = 100,
        device: str = "cpu"
    ):
        super().__init__(None, feature_names)
        self.predict_fn = predict_fn
        self.background_data = background_data
        self.device = torch.device(device)
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.explainer_net = FastSHAPNetwork(self.n_features, self.hidden_dims).to(self.device)
        self.training_time_seconds: float = 0.0
        self.is_trained = False

    def train_explainer(
        self,
        X_train: np.ndarray,
        teacher_attributions: Optional[np.ndarray] = None
    ) -> "FastSHAPExplainer":
        """Trains the explainer network using either teacher distillation or coalition learning."""
        start_t = time.perf_counter()
        torch.set_num_threads(1)
        logger.info(f"Training FastSHAP explainer on {len(X_train)} samples for {self.epochs} epochs...")
        
        optimizer = optim.Adam(self.explainer_net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()

        if teacher_attributions is not None:
            # Teacher distillation: learn directly from game-theoretic references (MSE + Cosine directional ranking)
            X_t = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
            Y_t = torch.from_numpy(teacher_attributions.astype(np.float32)).to(self.device)
            dataset = TensorDataset(X_t, Y_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(1, self.epochs + 1):
                self.explainer_net.train()
                epoch_loss = 0.0
                for bx, by in loader:
                    pred = self.explainer_net(bx)
                    mse = criterion(pred, by)
                    cos_loss = 1.0 - torch.nn.functional.cosine_similarity(pred, by, dim=1).mean()
                    loss = mse + 0.1 * cos_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(bx)
                scheduler.step()
                epoch_loss /= len(X_train)
                if epoch % 25 == 0 or epoch == self.epochs:
                    logger.info(f"FastSHAP (Distilled) Epoch [{epoch}/{self.epochs}] - Loss: {epoch_loss:.6f}")
        else:
            # Model-agnostic coalition training with stratified coalition sampling and efficiency constraint
            bg_preds = self.predict_fn(self.background_data)
            if bg_preds.ndim == 2:
                bg_preds = bg_preds[:, 1]
            base_val = float(np.mean(bg_preds))

            X_t = torch.from_numpy(X_train.astype(np.float32))
            dataset = TensorDataset(X_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            bg_tensor = torch.from_numpy(self.background_data.astype(np.float32))

            for epoch in range(1, self.epochs + 1):
                self.explainer_net.train()
                epoch_loss = 0.0
                for (bx,) in loader:
                    bx = bx.to(self.device)
                    B = len(bx)
                    
                    # Stratified coalition sampling
                    coalition = torch.zeros((B, self.n_features), device=self.device)
                    r = torch.rand(B, device=self.device)
                    for b in range(B):
                        if r[b] < 0.30:  # Single-feature coalitions
                            coalition[b, torch.randint(0, self.n_features, (1,))] = 1.0
                        elif r[b] < 0.60:  # Leave-one-out coalitions
                            coalition[b] = 1.0
                            coalition[b, torch.randint(0, self.n_features, (1,))] = 0.0
                        else:  # General Bernoulli coalitions
                            coalition[b] = torch.bernoulli(torch.full((self.n_features,), 0.5, device=self.device))

                    bg_idx = torch.randint(0, len(bg_tensor), (B,))
                    bg_sample = bg_tensor[bg_idx].to(self.device)
                    x_hybrid = coalition * bx + (1.0 - coalition) * bg_sample
                    
                    with torch.no_grad():
                        preds_hybrid = self.predict_fn(x_hybrid.cpu().numpy())
                        if preds_hybrid.ndim == 2:
                            preds_hybrid = preds_hybrid[:, 1]
                        f_val = torch.from_numpy(preds_hybrid.astype(np.float32)).to(self.device)

                        preds_full = self.predict_fn(bx.cpu().numpy())
                        if preds_full.ndim == 2:
                            preds_full = preds_full[:, 1]
                        f_full = torch.from_numpy(preds_full.astype(np.float32)).to(self.device)

                    phi = self.explainer_net(bx)
                    loss_coalition = criterion(torch.sum(phi * coalition, dim=1), f_val - base_val)
                    loss_efficiency = criterion(torch.sum(phi, dim=1), f_full - base_val)
                    loss = loss_coalition + 2.0 * loss_efficiency

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * B

                scheduler.step()
                epoch_loss /= len(X_train)
                if epoch % 25 == 0 or epoch == self.epochs:
                    logger.info(f"FastSHAP (Coalition) Epoch [{epoch}/{self.epochs}] - Loss: {epoch_loss:.6f}")

        self.training_time_seconds = time.perf_counter() - start_t
        self.is_trained = True
        logger.info(f"FastSHAP training completed in {self.training_time_seconds:.2f} seconds.")
        return self

    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("FastSHAP explainer must be trained before calling explain_instance.")
        self.explainer_net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x.reshape(1, -1).astype(np.float32)).to(self.device)
            phi = self.explainer_net(x_t).cpu().numpy()[0]
            return phi

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("FastSHAP explainer must be trained before calling explain_batch.")
        self.explainer_net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            phi = self.explainer_net(x_t).cpu().numpy()
            return phi

class LIMEExplainer(BaseExplainer):
    """
    Local Interpretable Model-agnostic Explainer (Ribeiro et al., KDD 2016).
    Builds a local ridge surrogate weighted by distance to the query instance.
    Attributions are scaled by feature deviation from the reference baseline
    to accurately reflect instance-level risk contributions.
    """
    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        training_data: np.ndarray,
        feature_names: List[str] = FEATURE_NAMES,
        num_samples: int = 500,
        random_state: int = 42
    ):
        super().__init__(None, feature_names)
        self.predict_fn = predict_fn
        self.num_samples = num_samples
        self.random_state = random_state
        self.training_data = training_data
        self.mean_train = np.mean(training_data, axis=0)
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=self.feature_names,
            class_names=["NoDefault", "Default"],
            mode="classification",
            discretize_continuous=False,
            random_state=self.random_state
        )

    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        exp = self.explainer.explain_instance(
            x,
            self._predict_wrapper,
            labels=(1,),
            num_features=self.n_features,
            num_samples=self.num_samples
        )
        attributions = np.zeros(self.n_features)
        label = 1 if 1 in exp.local_exp else list(exp.local_exp.keys())[0]
        for feat_idx, weight in exp.local_exp[label]:
            # Scale regression slope by the instance's deviation from reference mean
            attributions[feat_idx] = weight * (x[feat_idx] - self.mean_train[feat_idx])
        return attributions

    def explain_instance_with_score(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns feature attributions and local surrogate score (R^2)."""
        exp = self.explainer.explain_instance(
            x,
            self._predict_wrapper,
            labels=(1,),
            num_features=self.n_features,
            num_samples=self.num_samples
        )
        attributions = np.zeros(self.n_features)
        label = 1 if 1 in exp.local_exp else list(exp.local_exp.keys())[0]
        for feat_idx, weight in exp.local_exp[label]:
            attributions[feat_idx] = weight * (x[feat_idx] - self.mean_train[feat_idx])
        r2 = float(exp.score)
        return attributions, r2

    def _predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict_fn(X)
        if preds.ndim == 1:
            return np.column_stack([1.0 - preds, preds])
        return preds
