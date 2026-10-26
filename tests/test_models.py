import pytest
import numpy as np
from credit_xai.models import MLPCreditModel, evaluate_model
from credit_xai.data import FEATURE_NAMES

def test_mlp_model_training_and_inference():
    np.random.seed(42)
    X_train = np.random.randn(100, len(FEATURE_NAMES))
    y_train = np.random.binomial(1, 0.2, size=100)
    X_val = np.random.randn(30, len(FEATURE_NAMES))
    y_val = np.random.binomial(1, 0.2, size=30)

    mlp = MLPCreditModel(
        input_dim=len(FEATURE_NAMES),
        hidden_dims=[16, 8],
        epochs=3,
        batch_size=32,
        random_state=42
    )
    mlp.fit(X_train, y_train, X_val, y_val)
    
    probs = mlp.predict_proba(X_val)
    assert probs.shape == (30, 2)
    # Check probabilities are valid [0, 1] and sum to 1
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    metrics = evaluate_model(mlp, X_val, y_val)
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
