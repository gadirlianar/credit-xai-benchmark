from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from credit_xai.data import FEATURE_NAMES, TARGET_NAME
from credit_xai.utils import get_logger, save_json, ensure_dirs

logger = get_logger("credit_xai.preprocessing")

@dataclass
class PreprocessorState:
    medians: Dict[str, float] = field(default_factory=dict)
    scaler_means: Dict[str, float] = field(default_factory=dict)
    scaler_scales: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    outlier_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

class CreditDataPreprocessor:
    """
    Leakage-free preprocessing pipeline fitted exclusively on training data.
    Performs outlier winsorization, median imputation, and standard scaling.
    """
    def __init__(self, outlier_config: Optional[Dict[str, float]] = None):
        self.outlier_config = outlier_config or {
            "clip_revolving_utilization": 13.0,
            "clip_debt_ratio": 3000.0,
            "min_age": 18.0,
            "max_age": 100.0
        }
        self.state = PreprocessorState()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "CreditDataPreprocessor":
        logger.info("Fitting preprocessor on training partition...")
        X = df[FEATURE_NAMES].copy()
        
        # 1. Learn outlier bounds
        self.state.outlier_bounds = {
            "RevolvingUtilizationOfUnsecuredLines": (0.0, self.outlier_config.get("clip_revolving_utilization", 13.0)),
            "DebtRatio": (0.0, self.outlier_config.get("clip_debt_ratio", 3000.0)),
            "age": (self.outlier_config.get("min_age", 18.0), self.outlier_config.get("max_age", 100.0))
        }
        
        # Winsorize for median calculation
        for col, (lower, upper) in self.state.outlier_bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        
        # 2. Learn medians on training data
        for col in FEATURE_NAMES:
            self.state.medians[col] = float(X[col].median(skipna=True))
            X[col] = X[col].fillna(self.state.medians[col])

        # 3. Fit scaler
        self.scaler.fit(X[FEATURE_NAMES])
        self.state.feature_names = FEATURE_NAMES
        for i, col in enumerate(FEATURE_NAMES):
            self.state.scaler_means[col] = float(self.scaler.mean_[i])
            self.state.scaler_scales[col] = float(self.scaler.scale_[i])

        self.is_fitted = True
        logger.info("Preprocessor successfully fitted.")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform can be called.")
        
        X = df[FEATURE_NAMES].copy()
        # 1. Clip outliers based on fitted training bounds
        for col, (lower, upper) in self.state.outlier_bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)

        # 2. Impute missing values with learned medians
        for col in FEATURE_NAMES:
            X[col] = X[col].fillna(self.state.medians[col])

        # 3. Produce scaled matrix and unscaled DataFrame
        X_scaled = self.scaler.transform(X[FEATURE_NAMES])
        
        y = None
        if TARGET_NAME in df.columns:
            y = df[TARGET_NAME].values.astype(int)

        return X_scaled, y, X

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        return self.fit(df).transform(df)

    def save_state(self, path: Path) -> None:
        save_json({
            "medians": self.state.medians,
            "scaler_means": self.state.scaler_means,
            "scaler_scales": self.state.scaler_scales,
            "feature_names": self.state.feature_names,
            "outlier_bounds": {k: list(v) for k, v in self.state.outlier_bounds.items()}
        }, path)

def partition_and_preprocess(
    df: pd.DataFrame,
    output_dir: Path,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CreditDataPreprocessor]:
    """
    Stratified split into Train / Val / Test partitions, fits preprocessor
    strictly on Train, transforms all splits, and saves output artifacts.
    """
    output_dir = Path(output_dir)
    ensure_dirs(output_dir)

    logger.info(f"Performing stratified split: Train={train_split}, Val={val_split}, Test={test_split}")
    
    # Stratified split: Train vs Temp (Val + Test)
    temp_split = val_split + test_split
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_split,
        random_state=seed,
        stratify=df[TARGET_NAME]
    )
    
    # Split Temp into Val and Test
    val_ratio = val_split / temp_split
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio),
        random_state=seed,
        stratify=temp_df[TARGET_NAME]
    )

    logger.info(f"Partition sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    preprocessor = CreditDataPreprocessor()
    preprocessor.fit(train_df)
    preprocessor.save_state(output_dir / "preprocessor.json")

    # Save partitions
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    return train_df, val_df, test_df, preprocessor
