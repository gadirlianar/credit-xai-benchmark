from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from credit_xai.explainers import BaseExplainer
from credit_xai.data import FEATURE_NAMES
from credit_xai.utils import get_logger

logger = get_logger("credit_xai.faithfulness")

@dataclass
class FaithfulnessResult:
    model_name: str
    explainer_name: str
    configuration: str
    sample_count: int
    normalized_faithfulness: float
    normalized_faithfulness_ci_lower: float
    normalized_faithfulness_ci_upper: float
    normalized_deletion_auc: float
    raw_deletion_auc: float
    top1_impact_delta: float
    top3_impact_delta: float
    top5_impact_delta: float
    mean_spearman_rho: Optional[float] = None
    mean_kendall_tau: Optional[float] = None
    mean_l2_error: Optional[float] = None
    top1_rank_agreement: Optional[float] = None
    top3_rank_agreement: Optional[float] = None
    mean_surrogate_r2: Optional[float] = None

def compute_bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """Computes 95% bootstrap confidence interval for sample mean."""
    vals = np.array(values)
    boot_means = []
    rng = np.random.RandomState(42)
    n = len(vals)
    for _ in range(n_bootstrap):
        sample = rng.choice(vals, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2)))
    return round(lower, 4), round(upper, 4)

class FaithfulnessEvaluator:
    """
    Principled multi-metric explanation faithfulness evaluation harness.
    Evaluates prediction confidence degradation under feature deletion,
    top-k impact, scale-invariant normalized deletion curves, and direct
    alignment (Spearman rho, L2 error, Top-k agreement) with ground-truth references.
    """
    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
        feature_names: List[str] = FEATURE_NAMES,
        baseline_strategy: str = "median"
    ):
        self.predict_fn = predict_fn
        self.background_data = background_data
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
        # Calculate baseline replacement vector (median of background samples)
        if baseline_strategy == "mean":
            self.baseline_vector = np.mean(background_data, axis=0)
        else:
            self.baseline_vector = np.median(background_data, axis=0)

    def _get_prob(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict_fn(X)
        if preds.ndim == 2:
            return preds[:, 1]
        return preds.ravel()

    def compute_deletion_curve(self, x: np.ndarray, attributions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Sequentially removes features in descending order of attributed importance.
        Returns:
            - raw_curve: Absolute model prediction trajectory [C(0), C(1), ..., C(d)]
            - norm_curve: Scale-invariant normalized trajectory C_norm(k) in [0, 1]
            - raw_auc: Area under raw curve
            - norm_auc: Area under normalized curve (NAUDC)
        """
        orig_prob = self._get_prob(x.reshape(1, -1))[0]
        base_prob = self._get_prob(self.baseline_vector.reshape(1, -1))[0]
        
        # Sort features by descending absolute attribution magnitude
        sorted_indices = np.argsort(-np.abs(attributions))
        
        curve = [orig_prob]
        x_perturbed = x.copy()
        
        for idx in sorted_indices:
            x_perturbed[idx] = self.baseline_vector[idx]
            new_prob = self._get_prob(x_perturbed.reshape(1, -1))[0]
            curve.append(new_prob)
            
        curve_arr = np.array(curve)
        
        # Scale-invariant normalization: C_tilde(k) = (C(k) - C(d)) / (C(0) - C(d))
        denom = orig_prob - base_prob
        if abs(denom) > 1e-5:
            norm_curve = np.clip((curve_arr - base_prob) / denom, 0.0, 1.0)
        else:
            # Fallback for near-zero probability movement
            norm_curve = np.linspace(1.0, 0.0, len(curve_arr))

        try:
            raw_auc = float(np.trapezoid(curve_arr) / self.n_features)
            norm_auc = float(np.trapezoid(norm_curve) / self.n_features)
        except AttributeError:
            from scipy.integrate import trapezoid
            raw_auc = float(trapezoid(curve_arr) / self.n_features)
            norm_auc = float(trapezoid(norm_curve) / self.n_features)

        return curve_arr, norm_curve, raw_auc, norm_auc

    def evaluate_dataset_faithfulness(
        self,
        model_name: str,
        explainer_name: str,
        explainer: BaseExplainer,
        X_eval: np.ndarray,
        configuration: str = "default",
        reference_attributions: Optional[np.ndarray] = None
    ) -> FaithfulnessResult:
        logger.info(f"Evaluating faithfulness for {model_name} + {explainer_name} on {len(X_eval)} instances...")
        
        raw_aucs: List[float] = []
        norm_aucs: List[float] = []
        top1_deltas: List[float] = []
        top3_deltas: List[float] = []
        top5_deltas: List[float] = []
        spearman_rhos: List[float] = []
        kendall_taus: List[float] = []
        l2_errors: List[float] = []
        top1_agreements: List[float] = []
        top3_agreements: List[float] = []
        surrogate_r2s: List[float] = []

        for i in range(len(X_eval)):
            x = X_eval[i]
            
            # Get attributions and surrogate R^2 if available (e.g. from LIME)
            if hasattr(explainer, "explain_instance_with_score"):
                attr, r2 = explainer.explain_instance_with_score(x)
                if r2 is not None and not np.isnan(r2):
                    surrogate_r2s.append(r2)
            else:
                attr = explainer.explain_instance(x)

            # Deletion curve evaluation
            raw_curve, norm_curve, r_auc, n_auc = self.compute_deletion_curve(x, attr)
            raw_aucs.append(r_auc)
            norm_aucs.append(n_auc)
            
            orig_p = raw_curve[0]
            if len(raw_curve) > 1:
                top1_deltas.append(abs(orig_p - raw_curve[1]))
            if len(raw_curve) > 3:
                top3_deltas.append(abs(orig_p - raw_curve[3]))
            if len(raw_curve) > 5:
                top5_deltas.append(abs(orig_p - raw_curve[5]))

            # Direct alignment with reference attributions (e.g., TreeSHAP exact ground truth)
            if reference_attributions is not None:
                ref_attr = reference_attributions[i]
                rho, _ = spearmanr(attr, ref_attr)
                tau, _ = kendalltau(attr, ref_attr)
                if not np.isnan(rho):
                    spearman_rhos.append(rho)
                if not np.isnan(tau):
                    kendall_taus.append(tau)
                
                # L2 error (MSE) between attribution vectors
                l2 = float(np.mean((attr - ref_attr) ** 2))
                l2_errors.append(l2)

                # Top-1 exact agreement
                pred_top1 = np.argmax(np.abs(attr))
                ref_top1 = np.argmax(np.abs(ref_attr))
                top1_agreements.append(1.0 if pred_top1 == ref_top1 else 0.0)

                # Top-3 Jaccard overlap
                pred_top3 = set(np.argsort(-np.abs(attr))[:3])
                ref_top3 = set(np.argsort(-np.abs(ref_attr))[:3])
                jaccard_top3 = len(pred_top3 & ref_top3) / len(pred_top3 | ref_top3)
                top3_agreements.append(jaccard_top3)

        mean_spearman = float(np.mean(spearman_rhos)) if spearman_rhos else None
        mean_kendall = float(np.mean(kendall_taus)) if kendall_taus else None
        mean_l2 = float(np.mean(l2_errors)) if l2_errors else None
        top1_agr = float(np.mean(top1_agreements)) if top1_agreements else None
        top3_agr = float(np.mean(top3_agreements)) if top3_agreements else None
        mean_r2 = float(np.mean(surrogate_r2s)) if surrogate_r2s else None

        individual_faithfulness = [1.0 - a for a in norm_aucs]
        ci_lower, ci_upper = compute_bootstrap_ci(individual_faithfulness)
        mean_norm_auc = float(np.mean(norm_aucs))
        normalized_faith = round(1.0 - mean_norm_auc, 4)

        result = FaithfulnessResult(
            model_name=model_name,
            explainer_name=explainer_name,
            configuration=configuration,
            sample_count=len(X_eval),
            normalized_faithfulness=normalized_faith,
            normalized_faithfulness_ci_lower=ci_lower,
            normalized_faithfulness_ci_upper=ci_upper,
            normalized_deletion_auc=round(mean_norm_auc, 4),
            raw_deletion_auc=round(float(np.mean(raw_aucs)), 4),
            top1_impact_delta=round(float(np.mean(top1_deltas)), 4),
            top3_impact_delta=round(float(np.mean(top3_deltas)), 4),
            top5_impact_delta=round(float(np.mean(top5_deltas)), 4),
            mean_spearman_rho=round(mean_spearman, 4) if mean_spearman is not None else None,
            mean_kendall_tau=round(mean_kendall, 4) if mean_kendall is not None else None,
            mean_l2_error=round(mean_l2, 6) if mean_l2 is not None else None,
            top1_rank_agreement=round(top1_agr, 4) if top1_agr is not None else None,
            top3_rank_agreement=round(top3_agr, 4) if top3_agr is not None else None,
            mean_surrogate_r2=round(mean_r2, 4) if mean_r2 is not None else None
        )
        return result
