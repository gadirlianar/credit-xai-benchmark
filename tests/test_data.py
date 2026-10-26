import pytest
import numpy as np
import pandas as pd
from credit_xai.data import FEATURE_NAMES, TARGET_NAME, get_data_dictionary, generate_synthetic_gmsc

def test_feature_names_integrity():
    assert len(FEATURE_NAMES) == 10
    assert "DebtRatio" in FEATURE_NAMES
    assert "MonthlyIncome" in FEATURE_NAMES
    assert "RevolvingUtilizationOfUnsecuredLines" in FEATURE_NAMES

def test_data_dictionary():
    dd = get_data_dictionary()
    assert TARGET_NAME in dd
    assert dd[TARGET_NAME]["type"] == "binary"
    for f in FEATURE_NAMES:
        assert f in dd
        assert "description" in dd[f]

def test_synthetic_gmsc_generation():
    df = generate_synthetic_gmsc(n_samples=500, seed=42)
    assert len(df) == 500
    assert TARGET_NAME in df.columns
    assert set(FEATURE_NAMES).issubset(set(df.columns))
    # Test realistic default rate roughly in [0.03, 0.12]
    default_rate = df[TARGET_NAME].mean()
    assert 0.03 <= default_rate <= 0.12
