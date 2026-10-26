# Methodology: Latency vs. Explanation Faithfulness Benchmark

## 1. Experimental Motivation & Research Questions
In modern financial institutions and online credit underwriting, automated decision pipelines operate under strict Service Level Agreements (SLAs, e.g., p95 < 50ms or < 100ms). Concurrently, international legal frameworks such as the **EU AI Act (Regulation (EU) 2024/1689, Annex III section 5(b))** and **GDPR (Articles 13-15, 22)** mandate transparency and explainability for creditworthiness evaluation.

This benchmark rigorously evaluates the trade-offs between:
- **Computational Performance**: Component-wise inference latency ($L_{\text{pred}}$), explanation latency ($L_{\text{xai}}$), and end-to-end processing ($L_{\text{e2e}}$) across percentiles (p50, p95, p99).
- **Explanation Faithfulness**: How accurately feature attributions reflect the true underlying decision surface of the predictive models.
- **Robustness & Stability**: Invariance to stochastic seeds, background reference sizes, and bounded input perturbations.
- **Amortization Economics**: Evaluating the break-even query volume for learned explainers (FastSHAP) versus per-instance sampling (KernelSHAP).

## 2. Base Predictive Models
1. **XGBoost**: Gradient boosted decision tree ensemble with histogram binning and early stopping.
2. **LightGBM**: Highly optimized gradient boosted trees utilizing leaf-wise tree growth.
3. **Multilayer Perceptron (MLP)**: Feedforward deep neural network (2 hidden layers, ReLU activation, dropout, Adam optimizer) implemented in PyTorch.
4. **Logistic Regression (Interpretable Baseline)**: Generalized linear model with L2 regularization.

## 3. Explainability Frameworks
- **TreeSHAP**: Polynomial-time exact and interventional Shapley value computation exploiting tree path structures ($O(TLD^2)$).
- **KernelSHAP**: Model-agnostic weighted linear surrogate approximating Shapley values via coalition sampling ($O(2^M)$ or $M \cdot n_{\text{samples}}$).
- **FastSHAP**: Amortized neural explainer trained via an auxiliary neural network $\phi_\theta(x)$ optimizing the Shapley value loss $\mathcal{L}_{\text{Shapley}}$ with paired sampling. Yields $O(1)$ constant-time inference.
- **LIME**: Local Interpretable Model-agnostic Explanations fitting an interpretable ridge regressor in an exponential-kernel neighborhood around the query instance.

## 4. Evaluation Metrics
### 4.1 Latency & Resource Utilization
- **High-Resolution Monotonic Clock**: `time.perf_counter_ns()` with pre-benchmark warm-up runs ($N_{\text{warmup}} = 25$) to eliminate JIT and caching artifacts.
- Metrics recorded: Mean, Std, p50, p95, p99 (over $N = 1000$ timed runs), throughput (queries/sec), memory footprint (resident set size $\Delta \text{RSS}$).

### 4.2 Faithfulness Metrics
1. **Feature Deletion Curve (Area Under Deletion Curve - AUDC)**: Sequentially masking the top-$k$ most important features by setting them to the background baseline distribution and observing the degradation in prediction confidence. A faithful explainer causes rapid drop in model confidence.
2. **Rank Correlation against Ground-Truth Reference**: Spearman rank correlation $\rho_s$ and Kendall's $\tau$ between surrogate attributions and high-budget exact Shapley references.
3. **Local Surrogate Fidelity ($R^2$)**: Quantifying how well the local surrogate approximates the black-box model in the localized neighborhood.

### 4.3 Robustness & Stability
1. **Stochastic Invariance (Jaccard Similarity)**: Measuring top-$k$ feature set overlap across $S$ repeated runs with varied random seeds.
2. **Lipschitz Continuity / Perturbation Sensitivity**: Measuring the maximal change in the explanation vector normalized by small bounded input perturbations $\|\Delta x\| \le \epsilon$.

### 4.4 Pareto Frontier Analysis
Identifying non-dominated configurations in the 2D objective space:
$$\min (\text{Latency p95}) \quad \text{and} \quad \max (\text{Faithfulness AUDC})$$
Categorizing configurations into: Low-Latency, High-Faithfulness, Balanced Pareto Optimal, or Dominated.
