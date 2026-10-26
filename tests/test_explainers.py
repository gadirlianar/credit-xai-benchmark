import pytest
import numpy as np
from credit_xai.explainers import FastSHAPExplainer
from credit_xai.data import FEATURE_NAMES

def test_fastshap_explainer():
    np.random.seed(42)
    n_features = len(FEATURE_NAMES)
    X_train = np.random.randn(50, n_features)
    X_bg = np.random.randn(10, n_features)

    # Dummy predictor returning probability
    def dummy_predict_proba(X):
        p = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1.0 - p, p])

    explainer = FastSHAPExplainer(
        predict_fn=dummy_predict_proba,
        background_data=X_bg,
        hidden_dims=[32, 32],
        epochs=2,
        batch_size=16
    )
    explainer.train_explainer(X_train)
    assert explainer.is_trained

    # Explain single instance
    x = np.random.randn(n_features)
    attr = explainer.explain_instance(x)
    assert attr.shape == (n_features,)
    assert not np.isnan(attr).any()

    # Explain batch
    X_test = np.random.randn(5, n_features)
    batch_attr = explainer.explain_batch(X_test)
    assert batch_attr.shape == (5, n_features)
