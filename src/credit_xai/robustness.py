from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from scipy.stats import spearmanr

from credit_xai.explainers import BaseExplainer
from credit_xai.data import FEATURE_NAMES
from credit_xai.utils import get_logger

logger = get_logger("credit_xai.robustness")

@dataclass
class RobustnessResult:
    model_name: str
    explainer_name: str
    configuration: str
    top_k: int
    mean_jaccard_similarity: float
    mean_rank_correlation: float
    mean_cosine_similarity: float
    mean_lipschitz_drift: float
    perturbation_epsilon: float

class RobustnessEvaluator:
    """
    Evaluates explanation stability across random seeds, bounded input perturbations,
    and reference sample size variations.
    """
    def __init__(self, feature_names: List[str] = FEATURE_NAMES):
        self.feature_names = feature_names
        self.n_features = len(feature_names)

    @staticmethod
    def jaccard_similarity(set_a: set, set_b: set) -> float:
        union = len(set_a.union(set_b))
        if union == 0:
            return 1.0
        return len(set_a.intersection(set_b)) / union

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def evaluate_stochastic_stability(
        self,
        explainer_builder: Callable[[int], BaseExplainer],
        X_sample: np.ndarray,
        seeds: List[int] = [42, 123, 456, 789, 1011],
        top_k: int = 3
    ) -> Dict[str, float]:
        """
        Runs explainer with different random seeds on identical instances to measure repeatability.
        """
        logger.info(f"Testing stochastic repeatability across {len(seeds)} seeds on {len(X_sample)} instances...")
        all_attributions: List[np.ndarray] = []
        for s in seeds:
            explainer = explainer_builder(s)
            attrs = np.array([explainer.explain_instance(x) for x in X_sample])
            all_attributions.append(attrs)

        jaccards: List[float] = []
        rhos: List[float] = []
        cosines: List[float] = []

        n_instances = len(X_sample)
        n_seeds = len(seeds)

        for i in range(n_instances):
            for s1 in range(n_seeds):
                for s2 in range(s1 + 1, n_seeds):
                    attr1 = all_attributions[s1][i]
                    attr2 = all_attributions[s2][i]

                    top_indices1 = set(np.argsort(-np.abs(attr1))[:top_k])
                    top_indices2 = set(np.argsort(-np.abs(attr2))[:top_k])

                    jaccards.append(self.jaccard_similarity(top_indices1, top_indices2))
                    cosines.append(self.cosine_similarity(attr1, attr2))

                    rho, _ = spearmanr(attr1, attr2)
                    if not np.isnan(rho):
                        rhos.append(rho)

        return {
            "mean_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else 1.0,
            "mean_spearman": round(float(np.mean(rhos)), 4) if rhos else 1.0,
            "mean_cosine": round(float(np.mean(cosines)), 4) if cosines else 1.0
        }

    def evaluate_input_perturbation_sensitivity(
        self,
        explainer: BaseExplainer,
        X_sample: np.ndarray,
        epsilon: float = 0.05,
        top_k: int = 3
    ) -> Dict[str, float]:
        """
        Applies plausible bounded noise delta_x and calculates explanation drift (Lipschitz ratio).
        """
        logger.info(f"Evaluating perturbation sensitivity with epsilon={epsilon}...")
        lipschitz_ratios: List[float] = []
        jaccards: List[float] = []

        for x in X_sample:
            orig_attr = explainer.explain_instance(x)
            
            # Generate bounded Gaussian noise in standardized space
            noise = np.random.normal(0, epsilon, size=len(x))
            x_perturbed = x + noise
            
            perturbed_attr = explainer.explain_instance(x_perturbed)

            dx_norm = np.linalg.norm(noise)
            d_phi_norm = np.linalg.norm(perturbed_attr - orig_attr)

            if dx_norm > 1e-6:
                lipschitz_ratios.append(float(d_phi_norm / dx_norm))

            top_orig = set(np.argsort(-np.abs(orig_attr))[:top_k])
            top_pert = set(np.argsort(-np.abs(perturbed_attr))[:top_k])
            jaccards.append(self.jaccard_similarity(top_orig, top_pert))

        return {
            "mean_lipschitz_drift": round(float(np.mean(lipschitz_ratios)), 4) if lipschitz_ratios else 0.0,
            "mean_perturbed_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else 1.0
        }

    def evaluate_ood_attribution_shift(
        self,
        surrogate_explainer: BaseExplainer,
        ground_truth_explainer: BaseExplainer,
        X_sample: np.ndarray,
        shift_sigma: float = 3.0
    ) -> Dict[str, float]:
        """
        Applies macroeconomic out-of-distribution (OOD) covariate shift (e.g. extreme utilization and debt spikes)
        to evaluate attribution drift and hallucination risk of amortized surrogates compared to exact ground truth.
        """
        logger.info(f"Evaluating OOD attribution shift under +{shift_sigma} sigma covariate perturbation...")
        id_surr = surrogate_explainer.explain_batch(X_sample)
        id_gt = ground_truth_explainer.explain_batch(X_sample)
        
        id_rhos = [spearmanr(id_surr[i], id_gt[i])[0] for i in range(len(X_sample))]
        id_mse = [float(np.mean((id_surr[i] - id_gt[i]) ** 2)) for i in range(len(X_sample))]

        # Shift feature 0 (RevolvingUtilization) and feature 2 (DebtRatio) by +shift_sigma in standardized space
        X_ood = X_sample.copy()
        X_ood[:, 0] += shift_sigma
        X_ood[:, 2] += shift_sigma

        ood_surr = surrogate_explainer.explain_batch(X_ood)
        ood_gt = ground_truth_explainer.explain_batch(X_ood)

        ood_rhos = [spearmanr(ood_surr[i], ood_gt[i])[0] for i in range(len(X_sample))]
        ood_mse = [float(np.mean((ood_surr[i] - ood_gt[i]) ** 2)) for i in range(len(X_sample))]

        return {
            "in_dist_spearman": round(float(np.nanmean(id_rhos)), 4),
            "in_dist_mse": round(float(np.mean(id_mse)), 6),
            "ood_spearman": round(float(np.nanmean(ood_rhos)), 4),
            "ood_mse": round(float(np.mean(ood_mse)), 6),
            "spearman_degradation_pct": round((float(np.nanmean(id_rhos)) - float(np.nanmean(ood_rhos))) / max(float(np.nanmean(id_rhos)), 1e-4) * 100.0, 2)
        }
