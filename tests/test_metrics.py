import pytest
import numpy as np
from credit_xai.faithfulness import FaithfulnessEvaluator
from credit_xai.robustness import RobustnessEvaluator
from credit_xai.statistics import ParetoConfiguration, compute_pareto_frontier, compute_fastshap_amortization

def test_jaccard_and_cosine_similarity():
    rob = RobustnessEvaluator()
    set_a = {0, 1, 2}
    set_b = {1, 2, 3}
    jaccard = rob.jaccard_similarity(set_a, set_b)
    # Intersection is {1, 2} (size 2), Union is {0, 1, 2, 3} (size 4) => 2/4 = 0.5
    assert jaccard == 0.5

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    cos = rob.cosine_similarity(v1, v2)
    assert np.isclose(cos, 1.0)

def test_pareto_frontier_logic():
    c1 = ParetoConfiguration(
        model_name="M1", explainer_name="Fast", configuration="def",
        latency_p50_ms=1.0, latency_p95_ms=2.0, faithfulness_score=0.7,
        stability_score=0.9, throughput_qps=500.0, strict_sla_50ms=True, broad_sla_100ms=True
    )
    c2 = ParetoConfiguration(
        model_name="M1", explainer_name="Slow", configuration="def",
        latency_p50_ms=10.0, latency_p95_ms=20.0, faithfulness_score=0.9,
        stability_score=0.95, throughput_qps=50.0, strict_sla_50ms=True, broad_sla_100ms=True
    )
    c3 = ParetoConfiguration(
        model_name="M1", explainer_name="DominatedBad", configuration="def",
        latency_p50_ms=50.0, latency_p95_ms=100.0, faithfulness_score=0.5,
        stability_score=0.8, throughput_qps=10.0, strict_sla_50ms=False, broad_sla_100ms=False
    )
    evaluated = compute_pareto_frontier([c1, c2, c3])
    # c1 and c2 should be Pareto optimal, c3 should be dominated
    p_map = {c.explainer_name: c.is_pareto_optimal for c in evaluated}
    assert p_map["Fast"] == True
    assert p_map["Slow"] == True
    assert p_map["DominatedBad"] == False

def test_fastshap_amortization():
    res = compute_fastshap_amortization(
        fastshap_training_seconds=10.0,
        fastshap_latency_ms=1.0,
        baseline_latency_ms=101.0,
        baseline_name="KernelSHAP"
    )
    assert res["amortization_viable"] == True
    # delta is (101 - 1)/1000 = 0.1 sec per query. 10.0 / 0.1 = 100 queries
    assert res["break_even_queries"] == 100
