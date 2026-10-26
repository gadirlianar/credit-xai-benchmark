from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from credit_xai.utils import load_yaml

@dataclass
class ProjectConfig:
    name: str = "credit-xai-benchmark"
    seed: int = 42
    device: str = "cpu"

@dataclass
class DataConfig:
    dataset_name: str = "give_me_some_credit"
    target_column: str = "SeriousDlqin2yrs"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    raw_filename: str = "give_me_some_credit.csv"
    processed_train_filename: str = "train.parquet"
    processed_val_filename: str = "val.parquet"
    processed_test_filename: str = "test.parquet"
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    stratify: bool = True
    subsample_size: int = 50000
    outlier_handling: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelConfig:
    xgboost: Dict[str, Any] = field(default_factory=dict)
    lightgbm: Dict[str, Any] = field(default_factory=dict)
    mlp: Dict[str, Any] = field(default_factory=dict)
    logistic_regression: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExplainerConfig:
    background_size: int = 100
    treeshap: Dict[str, Any] = field(default_factory=dict)
    kernelshap: Dict[str, Any] = field(default_factory=dict)
    fastshap: Dict[str, Any] = field(default_factory=dict)
    lime: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkConfig:
    warmup_runs: int = 25
    timed_samples: int = 1000
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])
    strict_sla_ms: float = 50.0
    broad_sla_ms: float = 100.0

@dataclass
class FaithfulnessConfig:
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    perturbation_steps: int = 10
    baseline_strategy: str = "mean"
    evaluate_deletion_auc: bool = True

@dataclass
class RobustnessConfig:
    stochastic_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])
    input_perturbation_epsilons: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    background_sample_sizes: List[int] = field(default_factory=lambda: [25, 50, 100, 200])

@dataclass
class ExperimentConfig:
    project: ProjectConfig
    data: DataConfig
    models: ModelConfig
    explainers: ExplainerConfig
    benchmarking: BenchmarkConfig
    faithfulness: FaithfulnessConfig
    robustness: RobustnessConfig
    paths: Dict[str, str]

    @classmethod
    def from_yaml(cls, path: str = "configs/experiment.yaml") -> "ExperimentConfig":
        raw = load_yaml(path)
        return cls(
            project=ProjectConfig(**raw.get("project", {})),
            data=DataConfig(**raw.get("data", {})),
            models=ModelConfig(**raw.get("models", {})),
            explainers=ExplainerConfig(**raw.get("explainers", {})),
            benchmarking=BenchmarkConfig(**raw.get("benchmarking", {})),
            faithfulness=FaithfulnessConfig(**raw.get("faithfulness", {})),
            robustness=RobustnessConfig(**raw.get("robustness", {})),
            paths=raw.get("paths", {})
        )
