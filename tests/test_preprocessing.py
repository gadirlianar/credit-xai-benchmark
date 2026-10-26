import pytest
import numpy as np
import pandas as pd
from credit_xai.data import FEATURE_NAMES, TARGET_NAME, generate_synthetic_gmsc
from credit_xai.preprocessing import CreditDataPreprocessor

def test_preprocessor_fit_transform():
    df = generate_synthetic_gmsc(n_samples=200, seed=42)
    # Inject an extreme outlier
    df.loc[0, "DebtRatio"] = 99999.0
    
    prep = CreditDataPreprocessor()
    X_scaled, y, X_unscaled = prep.fit_transform(df)

    assert X_scaled.shape == (200, len(FEATURE_NAMES))
    assert y.shape == (200,)
    # Verify no NaN remains after preprocessing
    assert not np.isnan(X_scaled).any()
    # Verify clipping worked
    assert X_unscaled.loc[0, "DebtRatio"] <= 3000.0

def test_unfitted_transform_raises():
    prep = CreditDataPreprocessor()
    df = generate_synthetic_gmsc(n_samples=50, seed=42)
    with pytest.raises(ValueError):
        prep.transform(df)
