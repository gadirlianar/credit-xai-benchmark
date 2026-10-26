import time
import os
import psutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
import pandas as pd

from credit_xai.explainers import BaseExplainer
from credit_xai.utils import get_logger, save_json, ensure_dirs

logger = get_logger("credit_xai.benchmark")

@dataclass
class TimingObservation:
    experiment_id: str
    timestamp: float
    model_name: str
    explainer_name: str
    configuration: str
    sample_index: int
    batch_size: int
    pred_latency_ms: float
    xai_latency_ms: float
    e2e_latency_ms: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class BenchmarkSummary:
    model_name: str
    explainer_name: str
    configuration: str
    batch_size: int
    sample_count: int
    # Component 1: Prediction Latency (ms)
    pred_mean_ms: float
    pred_std_ms: float
    pred_p50_ms: float
    pred_p95_ms: float
    pred_p99_ms: float
    # Component 2: Explanation Latency (ms)
    xai_mean_ms: float
    xai_std_ms: float
    xai_p50_ms: float
    xai_p95_ms: float
    xai_p99_ms: float
    # Component 3: End-to-End Latency (ms)
    e2e_mean_ms: float
    e2e_std_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float
    throughput_qps: float
    strict_sla_50ms_pct: float
    broad_sla_100ms_pct: float
    memory_rss_delta_mb: float

class LatencyBenchmarkHarness:
    """
    Reproducible high-resolution benchmarking harness.
    Separates model inference from explanation generation and end-to-end processing.
    """
    def __init__(
        self,
        experiment_id: str = "exp_credit_xai",
        warmup_runs: int = 25,
        strict_sla_ms: float = 50.0,
        broad_sla_ms: float = 100.0
    ):
        self.experiment_id = experiment_id
        self.warmup_runs = warmup_runs
        self.strict_sla_ms = strict_sla_ms
        self.broad_sla_ms = broad_sla_ms
        self.observations: List[TimingObservation] = []

    def run_benchmark(
        self,
        model_name: str,
        explainer_name: str,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        explainer: BaseExplainer,
        X_test: np.ndarray,
        configuration: str = "default",
        timed_samples: int = 1000,
        batch_size: int = 1
    ) -> BenchmarkSummary:
        logger.info(
            f"Benchmarking {model_name} + {explainer_name} ({configuration}) "
            f"for {timed_samples} samples (batch_size={batch_size}, warmup={self.warmup_runs})..."
        )
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        # 1. Warm-up runs to eliminate JIT compilation and cold caches
        n_warm = min(self.warmup_runs, len(X_test))
        for i in range(n_warm):
            idx = i % len(X_test)
            if batch_size == 1:
                sample = X_test[idx]
                _ = predict_fn(sample.reshape(1, -1))
                _ = explainer.explain_instance(sample)
            else:
                batch = X_test[:batch_size]
                _ = predict_fn(batch)
                _ = explainer.explain_batch(batch)

        # 2. Main timed iterations
        pred_times_ms: List[float] = []
        xai_times_ms: List[float] = []
        e2e_times_ms: List[float] = []

        n_iter = timed_samples if batch_size == 1 else (timed_samples // batch_size)
        start_wall_clock = time.perf_counter()

        for i in range(n_iter):
            idx = i % len(X_test)
            t_start_e2e = time.perf_counter_ns()
            
            try:
                if batch_size == 1:
                    sample = X_test[idx]
                    # Time Component 1: Prediction
                    t0 = time.perf_counter_ns()
                    pred = predict_fn(sample.reshape(1, -1))
                    t1 = time.perf_counter_ns()
                    pred_ms = (t1 - t0) / 1e6
                    
                    # Time Component 2: Explanation
                    t2 = time.perf_counter_ns()
                    exp = explainer.explain_instance(sample)
                    t3 = time.perf_counter_ns()
                    xai_ms = (t3 - t2) / 1e6
                else:
                    batch = X_test[idx : min(idx + batch_size, len(X_test))]
                    t0 = time.perf_counter_ns()
                    pred = predict_fn(batch)
                    t1 = time.perf_counter_ns()
                    pred_ms = ((t1 - t0) / 1e6) / len(batch)

                    t2 = time.perf_counter_ns()
                    exp = explainer.explain_batch(batch)
                    t3 = time.perf_counter_ns()
                    xai_ms = ((t3 - t2) / 1e6) / len(batch)

                t_end_e2e = time.perf_counter_ns()
                e2e_ms = (t_end_e2e - t_start_e2e) / 1e6
                if batch_size > 1:
                    e2e_ms /= len(batch)

                pred_times_ms.append(pred_ms)
                xai_times_ms.append(xai_ms)
                e2e_times_ms.append(e2e_ms)

                self.observations.append(
                    TimingObservation(
                        experiment_id=self.experiment_id,
                        timestamp=time.time(),
                        model_name=model_name,
                        explainer_name=explainer_name,
                        configuration=configuration,
                        sample_index=i,
                        batch_size=batch_size,
                        pred_latency_ms=pred_ms,
                        xai_latency_ms=xai_ms,
                        e2e_latency_ms=e2e_ms,
                        success=True
                    )
                )
            except Exception as e:
                logger.error(f"Error on instance {i}: {e}")
                self.observations.append(
                    TimingObservation(
                        experiment_id=self.experiment_id,
                        timestamp=time.time(),
                        model_name=model_name,
                        explainer_name=explainer_name,
                        configuration=configuration,
                        sample_index=i,
                        batch_size=batch_size,
                        pred_latency_ms=0.0,
                        xai_latency_ms=0.0,
                        e2e_latency_ms=0.0,
                        success=False,
                        error_message=str(e)
                    )
                )

        total_wall_sec = time.perf_counter() - start_wall_clock
        total_queries = len(pred_times_ms) * (batch_size if batch_size > 1 else 1)
        throughput = total_queries / max(total_wall_sec, 1e-6)

        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_delta = max(0.0, mem_after - mem_before)

        xai_arr = np.array(xai_times_ms)
        pred_arr = np.array(pred_times_ms)
        e2e_arr = np.array(e2e_times_ms)

        summary = BenchmarkSummary(
            model_name=model_name,
            explainer_name=explainer_name,
            configuration=configuration,
            batch_size=batch_size,
            sample_count=len(xai_arr),
            pred_mean_ms=round(float(np.mean(pred_arr)), 3),
            pred_std_ms=round(float(np.std(pred_arr)), 3),
            pred_p50_ms=round(float(np.percentile(pred_arr, 50)), 3),
            pred_p95_ms=round(float(np.percentile(pred_arr, 95)), 3),
            pred_p99_ms=round(float(np.percentile(pred_arr, 99)), 3),
            xai_mean_ms=round(float(np.mean(xai_arr)), 3),
            xai_std_ms=round(float(np.std(xai_arr)), 3),
            xai_p50_ms=round(float(np.percentile(xai_arr, 50)), 3),
            xai_p95_ms=round(float(np.percentile(xai_arr, 95)), 3),
            xai_p99_ms=round(float(np.percentile(xai_arr, 99)), 3),
            e2e_mean_ms=round(float(np.mean(e2e_arr)), 3),
            e2e_std_ms=round(float(np.std(e2e_arr)), 3),
            e2e_p50_ms=round(float(np.percentile(e2e_arr, 50)), 3),
            e2e_p95_ms=round(float(np.percentile(e2e_arr, 95)), 3),
            e2e_p99_ms=round(float(np.percentile(e2e_arr, 99)), 3),
            throughput_qps=round(float(throughput), 2),
            strict_sla_50ms_pct=round(float(np.mean(e2e_arr <= self.strict_sla_ms) * 100), 2),
            broad_sla_100ms_pct=round(float(np.mean(e2e_arr <= self.broad_sla_ms) * 100), 2),
            memory_rss_delta_mb=round(float(mem_delta), 2)
        )
        return summary

    def save_raw_observations(self, output_path: Path) -> None:
        """Saves individual timed queries to CSV."""
        ensure_dirs(output_path.parent)
        df = pd.DataFrame([asdict(obs) for obs in self.observations])
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} raw benchmark timing observations to {output_path}")
