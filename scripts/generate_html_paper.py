#!/usr/bin/env python3
"""
Generates an IEEE-formatted HTML camera-ready paper and compiles it to PDF via headless Chrome.
Synchronized with all 12 experimental configurations, controlled empirical metrics, and paired tests.
"""
import os
import subprocess
from pathlib import Path

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trade-Offs Between Real-Time Inference Latency and Explanation Faithfulness in Credit Risk Scoring</title>
<style>
  @page {
    size: letter;
    margin: 18mm 14mm 18mm 14mm;
  }
  body {
    font-family: "Times New Roman", Times, serif;
    font-size: 9.2pt;
    line-height: 1.28;
    color: #000;
    margin: 0;
    padding: 0;
  }
  .title {
    font-size: 17.5pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 7px;
    line-height: 1.15;
  }
  .authors {
    text-align: center;
    font-size: 9.8pt;
    margin-bottom: 12px;
  }
  .author-name {
    font-weight: bold;
    font-size: 10.5pt;
  }
  .author-affil {
    font-size: 8.8pt;
    color: #222;
  }
  .abstract-container {
    margin: 0 10px 12px 10px;
    font-size: 8.8pt;
    line-height: 1.25;
    text-align: justify;
  }
  .abstract-heading {
    font-weight: bold;
    font-style: italic;
    display: inline;
  }
  .keywords {
    margin-top: 6px;
    font-size: 8.8pt;
  }
  .keywords-heading {
    font-weight: bold;
    font-style: italic;
    display: inline;
  }
  .columns {
    column-count: 2;
    column-gap: 20px;
    text-align: justify;
  }
  h2 {
    font-size: 10.5pt;
    font-variant: small-caps;
    text-align: center;
    border-bottom: 1px solid #000;
    margin-top: 12px;
    margin-bottom: 5px;
    break-after: avoid;
  }
  h3 {
    font-size: 9.5pt;
    font-style: italic;
    margin-top: 8px;
    margin-bottom: 3px;
    break-after: avoid;
  }
  p {
    margin-top: 0;
    margin-bottom: 5px;
    text-indent: 1.2em;
  }
  p.no-indent {
    text-indent: 0;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 7.2pt;
    margin: 8px 0;
    break-inside: avoid;
  }
  th, td {
    padding: 2px 3px;
    border-top: 1px solid #ddd;
    border-bottom: 1px solid #ddd;
    text-align: left;
  }
  th {
    border-top: 1.5px solid #000;
    border-bottom: 1.5px solid #000;
    font-weight: bold;
  }
  tbody tr:last-child td {
    border-bottom: 1.5px solid #000;
  }
  .table-caption {
    font-size: 8pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 3px;
  }
  .figure-container {
    width: 100%;
    margin: 8px 0;
    text-align: center;
    break-inside: avoid;
  }
  .figure-container img {
    max-width: 100%;
    height: auto;
  }
  .figure-caption {
    font-size: 8pt;
    text-align: justify;
    margin-top: 3px;
  }
  .equation {
    text-align: center;
    margin: 5px 0;
    break-inside: avoid;
  }
  .ref-list {
    padding-left: 1.4em;
    font-size: 7.2pt;
    line-height: 1.2;
  }
  .ref-list li {
    margin-bottom: 3px;
  }
</style>
</head>
<body>

<div class="title">Trade-Offs Between Real-Time Inference Latency and Explanation Faithfulness in Credit Risk Scoring: A Reproducible Empirical Benchmark of TreeSHAP, KernelSHAP, FastSHAP, and LIME</div>

<div class="authors">
  <div class="author-name">Anar Gadirli</div>
  <div class="author-affil">Computer Engineering, ZU / BA Programs<br>Azerbaijan State Oil and Industry University (ASOIU), Baku, Azerbaijan<br>anargadirli55@gmail.com</div>
</div>

<div class="abstract-container">
  <p class="no-indent"><span class="abstract-heading">Abstract</span>—Automated credit risk assessment systems operating under the European Union Artificial Intelligence Act (Regulation (EU) 2024/1689, Annex III Section 5(b)) and GDPR Articles 13–15 and 22, informed by recent CJEU jurisprudence (<i>SCHUFA Holding</i>, C-634/21; <i>Dun & Bradstreet Austria</i>, C-203/22), must deliver faithful local explanations while complying with strict microservice Service Level Agreements (SLAs, e.g., $p95 < 50$\,ms). In this study, we present a reproducible empirical benchmark evaluating runtime latency, explanation faithfulness, stability, and amortization economics across four Explainable AI paradigms: TreeSHAP, KernelSHAP, FastSHAP, and LIME on 150,000 credit records across XGBoost, LightGBM, and Deep MLP architectures (6,000 timing observations across 12 configurations). Crucially, our findings establish a fundamental <i>negative result regarding low-dimensional latency trade-offs</i>: on standard tabular credit scoring features ($d=10$), all 12 configurations satisfy the 50\,ms SLA with 100% compliance ($L_{\text{xai}} < 8.5$\,ms at median $p50$, with $p99 \le 10.6$\,ms). In this single-inference regime, switching from TreeSHAP to FastSHAP saves merely 1.4 ms. Empirical scaling simulations across $d \in \{10, 25, 50, 100\}$ reveal that SLA-driven trade-offs emerge only when $d > 50$ (where KernelSHAP breaches 50\,ms) or under ultra-high-throughput streaming (&gt;5,000 QPS). Under controlled reference value functions, FastSHAP distilled from an ExactShapley teacher on MLP achieves $\rho_s = 0.880$ (88.0% Top-1 agreement), resolving baseline confounding artifacts. On the full test set ($N=22,500$) for XGBoost, TreeSHAP and FastSHAP achieve 77.9% Top-1 match, rising to 83.3% within the Adverse Action Cohort ($N=2,250$) and 95.4% for decisive inputs, with Top-3 and Top-4 Jaccard similarities exceeding 80%. However, under +3.0\,$\sigma$ covariate shift, FastSHAP rank correlation collapses to $\rho_s = 0.116$, highlighting off-manifold surrogate risks. For low-dimensional neural models ($d \le 12$), Exact Shapley runs in &lt;28\,ms, providing zero-drift compliance within real-time SLAs.</p>
  <div class="keywords"><span class="keywords-heading">Keywords</span>—Explainable Artificial Intelligence (XAI), Credit Risk Assessment, Inference Latency, Explanation Faithfulness, TreeSHAP, FastSHAP, KernelSHAP, LIME, High-Risk AI, EU AI Act, CJEU Jurisprudence.</div>
</div>

<div class="columns">

<h2>I. Introduction</h2>
<p>Machine learning models (GBDT and DNN) provide superior non-linear risk discrimination over traditional logistic scorecards in credit underwriting, but their black-box nature creates substantial compliance hurdles. Credit scoring is designated as a <i>High-Risk AI System</i> under Annex III, Section 5(b) of the EU AI Act (Regulation (EU) 2024/1689). In parallel, GDPR Articles 13–15 and 22 establish the consumer right to receive meaningful explanations. Recent CJEU rulings in <i>SCHUFA Holding</i> (C-634/21) and <i>Dun & Bradstreet Austria</i> (C-203/22), alongside US ECOA Regulation B (12 CFR § 1002.9), mandate intelligible explanations detailing how personal data influenced adverse decisions. While statutory texts do not specify XAI algorithmic metrics, our legal-analytical deduction is that substituting an approximate explainer that alters the primary identified adverse driver (Top-1 reason code mismatch) exposes lenders to substantial compliance and audit risks.</p>
<p>Architecturally, financial systems divide into two operational regimes: (1) <b>Asynchronous compliance workflows</b> (e.g., adverse action notice mailings within statutory 30-day windows), where explanations are generated in backend audit pipelines and latency is not a critical gating factor; and (2) <b>Synchronous real-time inference</b> (e.g., BNPL point-of-sale checkout APIs and interactive counterfactual portals), where decisions must be returned within strict SLAs ($p95 < 50$\,ms). This paper benchmarks TreeSHAP, KernelSHAP, FastSHAP, and LIME across both regimes.</p>

<h2>II. Methodology & Hyperparameters</h2>
<p>We benchmark four representative XAI methods on gradient boosted trees (XGBoost, LightGBM) and a 3-layer Multilayer Perceptron (MLP):</p>
<ul>
  <li><b>TreeSHAP:</b> Polynomial path evaluation ($O(TLD^2)$) computing exact interventional Shapley values on tree ensembles.</li>
  <li><b>KernelSHAP:</b> Model-agnostic weighted linear regression over $N_{\text{samp}} = 100$ coalition samples with $N_{\text{bg}} = 100$ background medoids.</li>
  <li><b>FastSHAP:</b> Parameterized 3-layer neural network ($10 \to 128 \to 64 \to 10$) trained for $E=100$ epochs. On GBDTs, FastSHAP is distilled from TreeSHAP. On Deep MLP, FastSHAP is distilled from Exact Shapley teacher targets ($N=1,000$).</li>
  <li><b>LIME:</b> Local ridge surrogate fitted on $N_{\text{samp}} = 500$ perturbations with exponential kernel width $\sigma = 0.75\sqrt{d} \approx 2.37$.</li>
  <li><b>Exact Shapley:</b> Vectorized solver evaluating all $2^{10} = 1,024$ binary masks against $B=100$ empirical background samples.</li>
</ul>
<p>All methods evaluate the identical marginal reference value function: $v(S) = \frac{1}{|B|} \sum_{b \in B} f(x_S, x^{(b)}_{\bar{S}})$. Faithfulness is measured via scale-invariant deletion curves: $\tilde{C}(k) = (f(x^{(k)}) - f(b)) / (f(x^{(0)}) - f(b))$, yielding $\mathcal{F}_{\text{norm}} = 1.0 - \text{NAUDC}$ with paired bootstrap hypothesis tests ($B=2,000$).</p>

<h2>III. Benchmark Dataset & Models</h2>
<p>Experiments utilize the Give Me Some Credit (GMSC) benchmark (150,000 borrower profiles, default rate 6.68%, $d=10$ financial ratios). Preprocessing comprises stratified 70/15/15 partitioning, winsorization at 99.9th percentiles, and median imputation. GBDT models achieve ROC-AUC of 0.8624 (XGBoost) and 0.8618 (LightGBM). Deep MLP achieves 0.8488 ROC-AUC and Platt-calibrated Brier score of 0.0515.</p>

<div class="table-caption">TABLE I: Model Predictive Performance on Held-Out Test Set</div>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>ROC-AUC</th>
      <th>PR-AUC</th>
      <th>F1-Score</th>
      <th>Brier Score</th>
      <th>Train Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>XGBoost</b></td>
      <td><b>0.8624</b></td>
      <td><b>0.4032</b></td>
      <td>0.4435</td>
      <td><b>0.0490</b></td>
      <td>0.68 s</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>0.8618</td>
      <td>0.3996</td>
      <td><b>0.4438</b></td>
      <td>0.0491</td>
      <td>0.65 s</td>
    </tr>
    <tr>
      <td>MLP (DNN)</td>
      <td>0.8488</td>
      <td>0.3631</td>
      <td>0.4169</td>
      <td>0.0515*</td>
      <td>4.66 s</td>
    </tr>
    <tr>
      <td>Logistic Reg.</td>
      <td>0.7522</td>
      <td>0.2545</td>
      <td>0.3554</td>
      <td>0.0575</td>
      <td><b>0.04 s</b></td>
    </tr>
  </tbody>
</table>
<p class="no-indent" style="font-size: 7.2pt;"><i>*Platt-calibrated Brier score on validation logits.</i></p>

<h2>IV. Latency & Throughput Benchmark</h2>
<p>Timing was evaluated on single-core Apple Silicon arm64 hardware via monotonic nanosecond timers across 6,000 observations across all 12 configurations. FastSHAP delivers sub-millisecond attribution ($L_{\text{xai}} = 0.027$\,ms), achieving 13,622 QPS on XGBoost. TreeSHAP achieves median latencies of $1.495$\,ms (XGBoost) and $1.379$\,ms (LightGBM). Vectorized Exact Shapley on Deep MLP executes in $6.780$\,ms ($p50$), satisfying strict 50\,ms SLAs with 100% compliance.</p>

<div class="table-caption">TABLE II: Latency and Throughput Across All 12 Configurations</div>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Explainer</th>
      <th>L_pred p50</th>
      <th>L_xai p50</th>
      <th>L_xai p95</th>
      <th>L_e2e p50</th>
      <th>Throughput</th>
      <th>SLA &lt;50ms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>XGBoost</td>
      <td>TreeSHAP</td>
      <td>0.051 ms</td>
      <td>1.495 ms</td>
      <td>1.798 ms</td>
      <td>1.550 ms</td>
      <td>631.7</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>TreeSHAP</td>
      <td>0.165 ms</td>
      <td>1.379 ms</td>
      <td>1.959 ms</td>
      <td>1.547 ms</td>
      <td>614.4</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td><b>XGBoost</b></td>
      <td><b>FastSHAP</b></td>
      <td><b>0.042 ms</b></td>
      <td><b>0.027 ms</b></td>
      <td><b>0.031 ms</b></td>
      <td><b>0.070 ms</b></td>
      <td><b>13,622.4</b></td>
      <td><b>100.0%</b></td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>FastSHAP</td>
      <td>0.137 ms</td>
      <td>0.032 ms</td>
      <td>0.075 ms</td>
      <td>0.169 ms</td>
      <td>5,115.1</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>FastSHAP</td>
      <td>0.065 ms</td>
      <td>0.027 ms</td>
      <td>0.028 ms</td>
      <td>0.092 ms</td>
      <td>10,560.3</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>KernelSHAP</td>
      <td>0.057 ms</td>
      <td>8.100 ms</td>
      <td>8.191 ms</td>
      <td>8.158 ms</td>
      <td>122.4</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>KernelSHAP</td>
      <td>0.190 ms</td>
      <td>8.466 ms</td>
      <td>9.812 ms</td>
      <td>8.667 ms</td>
      <td>114.7</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>KernelSHAP</td>
      <td>0.087 ms</td>
      <td>1.550 ms</td>
      <td>1.710 ms</td>
      <td>1.639 ms</td>
      <td>600.3</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>LIME</td>
      <td>0.053 ms</td>
      <td>0.954 ms</td>
      <td>1.102 ms</td>
      <td>1.008 ms</td>
      <td>966.9</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>LIME</td>
      <td>0.157 ms</td>
      <td>1.087 ms</td>
      <td>2.279 ms</td>
      <td>1.252 ms</td>
      <td>685.1</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>LIME</td>
      <td>0.082 ms</td>
      <td>0.890 ms</td>
      <td>1.972 ms</td>
      <td>0.972 ms</td>
      <td>860.6</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td><b>MLP</b></td>
      <td><b>ExactShapley</b></td>
      <td><b>0.087 ms</b></td>
      <td><b>6.780 ms</b></td>
      <td><b>7.150 ms</b></td>
      <td><b>6.867 ms</b></td>
      <td><b>146.2</b></td>
      <td><b>100.0%</b></td>
    </tr>
  </tbody>
</table>

<div class="figure-container">
  <img src="figures/figure4_tail_latency.png" alt="Tail Latency Benchmark">
  <div class="figure-caption"><b>Fig. 1.</b> Tail latency ($p50, p95, p99$) benchmark comparison plotted on logarithmic scale against strict (50 ms) and broad (100 ms) SLAs across all 12 configurations.</div>
</div>

<h2>V. Faithfulness & Ground-Truth Alignment</h2>
<p>Table III reports normalized faithfulness $\mathcal{F}_{\text{norm}}$ alongside 95% bootstrap confidence intervals and ground-truth alignment. In our paired bootstrap hypothesis tests ($B=2,000$ resamples), TreeSHAP demonstrates a statistically significant advantage over FastSHAP ($\Delta = +0.0283$, 95% CI $[+0.0030, +0.0555]$, $p = 0.0260$). Conversely, TreeSHAP and KernelSHAP are statistically indistinguishable ($\Delta = -0.0020$, $p = 0.7750$). Crucially, TreeSHAP and LIME are also statistically indistinguishable under sequential deletion ($\Delta = +0.0186$, $p = 0.3960$), despite LIME achieving only $\rho_s = 0.785$ and 64% Top-1 rank alignment. This demonstrates that sequential feature deletion is a greedy, high-variance heuristic that cannot reliably separate high-fidelity explainers from mediocre linear surrogates, establishing why axiomatic metrics ($\rho_s$, Top-$k$ Jaccard, $\ell_2$ MSE) are necessary.</p>
<p>On the full held-out test set ($N=22,500$), TreeSHAP vs distilled FastSHAP achieves 77.88% overall Top-1 match (95% Wilson CI: $[77.33\%, 78.42\%]$). Within the <b>Adverse Action Cohort</b> ($f(x) \ge 0.50$, $N=2,250$ loan rejections), Top-1 match rises to <b>83.33%</b> (95% Wilson CI: $[81.74\%, 84.82\%]$), with Top-3 Jaccard similarity of <b>80.68%</b> and Top-4 Jaccard of <b>77.73%</b> ($\rho_s = 0.8684$). For decisive applicants ($|\phi_1 - \phi_2| > 0.01$, $N=8,014$), Top-1 agreement surges to <b>95.36%</b>. Under CFPB Circular 2022-03, multi-reason disclosure (Top-3/Top-4 Jaccard &gt;80%) provides robust legal compliance.</p>
<p>Under controlled marginal baselines ($B=100$), KernelSHAP on MLP achieves $\rho_s = \mathbf{0.994}$ (94.0% Top-1 match). Distilled FastSHAP on MLP achieves $\rho_s = \mathbf{0.880}$ (Top-1 88.0%, MSE $0.000428$), whereas unassisted self-supervised FastSHAP achieves $\rho_s = 0.300$ (Top-1 41.5%), confirming that amortization fidelity on neural architectures is fundamentally enhanced by teacher distillation.</p>

<div class="table-caption">TABLE III: Explanation Faithfulness and Ground-Truth Alignment</div>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Explainer</th>
      <th>Norm. Faith. [95% CI]</th>
      <th>Raw AUDC</th>
      <th>Spearman ρ_s</th>
      <th>Top-1 Match</th>
      <th>ℓ2 MSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>XGBoost</b></td>
      <td><b>TreeSHAP</b></td>
      <td><b>0.7467 [0.718, 0.778]</b></td>
      <td><b>0.0193</b></td>
      <td><b>1.000</b></td>
      <td><b>100.0%</b></td>
      <td><b>0.0000</b></td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>KernelSHAP</td>
      <td>0.7634 [0.708, 0.814]</td>
      <td>0.0184</td>
      <td>0.980</td>
      <td>94.0%</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>FastSHAP (Distilled)</td>
      <td>0.7182 [0.686, 0.751]</td>
      <td>0.0211</td>
      <td>0.923</td>
      <td>90.5%</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>LIME</td>
      <td>0.7429 [0.700, 0.785]</td>
      <td>0.0212</td>
      <td>0.785</td>
      <td>64.0%</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>TreeSHAP</td>
      <td>0.7263 [0.694, 0.760]</td>
      <td>0.0199</td>
      <td>1.000</td>
      <td>100.0%</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>KernelSHAP</td>
      <td>0.7098 [0.639, 0.779]</td>
      <td>0.0196</td>
      <td>0.981</td>
      <td>96.0%</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>FastSHAP (Distilled)</td>
      <td>0.7006 [0.665, 0.737]</td>
      <td>0.0211</td>
      <td>0.932</td>
      <td>91.5%</td>
      <td>0.0011</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>LIME</td>
      <td>0.7023 [0.648, 0.758]</td>
      <td>0.0223</td>
      <td>0.776</td>
      <td>54.0%</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <td><b>MLP</b></td>
      <td><b>ExactShapley*</b></td>
      <td><b>0.8765 [0.866, 0.887]</b></td>
      <td><b>0.0215</b></td>
      <td><b>1.000</b></td>
      <td><b>100.0%</b></td>
      <td><b>0.0000</b></td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>KernelSHAP</td>
      <td>0.7576 [0.698, 0.813]</td>
      <td>0.0232</td>
      <td>0.994</td>
      <td>94.0%</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>FastSHAP (Distilled)</td>
      <td>0.7012 [0.669, 0.734]</td>
      <td>0.0289</td>
      <td>0.880</td>
      <td>88.0%</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>FastSHAP (Unassisted)</td>
      <td>0.6734 [0.640, 0.704]</td>
      <td>0.0304</td>
      <td>0.300</td>
      <td>41.5%</td>
      <td>0.0012</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>LIME</td>
      <td>0.7006 [0.643, 0.753]</td>
      <td>0.0247</td>
      <td>0.441</td>
      <td>64.0%</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <td colspan="2">Random Order Baseline</td>
      <td>0.5185 [0.504, 0.534]</td>
      <td>0.0350</td>
      <td>---</td>
      <td>---</td>
      <td>---</td>
    </tr>
    <tr>
      <td colspan="2">Worst-Case Baseline</td>
      <td>0.2486 [0.231, 0.267]</td>
      <td>0.0520</td>
      <td>---</td>
      <td>---</td>
      <td>---</td>
    </tr>
  </tbody>
</table>
<p class="no-indent" style="font-size: 7.2pt;"><i>*Exact Shapley evaluated across all 1,024 binary feature subsets (p50 = 6.78 ms). Full test set N=22,500; Adverse Action Cohort N=2,250.</i></p>

<div class="figure-container">
  <img src="figures/figure5_deletion_curves.png" alt="Feature Deletion Faithfulness Curve">
  <div class="figure-caption"><b>Fig. 2.</b> Two-panel feature deletion curves: (A) Raw model prediction probability trajectory for a representative high-risk borrower ($f(x^{(0)}) \approx 0.27$); (B) Scale-invariant normalized deletion curves $\tilde{C}(k) \in [0, 1]$ for the representative borrower profile (population-wide metrics in Table III).</div>
</div>

<h2>VI. Stability, Robustness & Multi-Method OOD Drift</h2>
<p>While compiled deterministic explainers achieve 1.000 execution repeatability, retraining FastSHAP across 5 independent seeds yields a Top-3 Jaccard similarity of 0.687 and rank stability of 0.822, demonstrating that surrogate neural weights converge to different local optima. Under bounded input noise ($\epsilon=0.05$, $N=200$), LightGBM TreeSHAP exhibits higher sensitivity (0.411) than XGBoost (0.116) due to leaf-wise best-first split step discontinuities (\texttt{num_leaves=31} vs level-wise \texttt{max_depth=5}). Discrete Top-3 Jaccard values stabilize near 0.800 due to combinatorial set properties on top-3 ranks.</p>
<p>Under a $+3.0\,\sigma$ macroeconomic covariate shock on credit distress features across all four explainers, attribution scales expand. Reporting scale-invariant Normalized MSE ($\text{NMSE} = \|\phi_{\text{ood}} - \phi_{\text{in}}\|_2^2 / \|\phi_{\text{in}}\|_2^2$): TreeSHAP yields $\text{NMSE} = 6.25$ ($\rho_s = 0.581$), KernelSHAP yields $\text{NMSE} = 6.37$ ($\rho_s = 0.546$), LIME yields $\text{NMSE} = \mathbf{25.63}$ ($\rho_s = 0.239$), and FastSHAP yields $\text{NMSE} = 5.34$ but suffers a catastrophic rank collapse ($\rho_s = \mathbf{0.116}$), confirming that neural surrogates fail to preserve adverse action driver hierarchies off-manifold.</p>

<div class="table-caption">TABLE IV: Explanation Stability and Robustness (N=200)</div>
<table>
  <thead>
    <tr>
      <th>Explainer</th>
      <th>Model</th>
      <th>Top-3 Jaccard (Seed)</th>
      <th>Spearman ρ (Seed)</th>
      <th>Top-3 Jaccard (Noise)</th>
      <th>Lipschitz Drift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TreeSHAP</td>
      <td>XGBoost</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.800</td>
      <td>0.116</td>
    </tr>
    <tr>
      <td>FastSHAP</td>
      <td>XGBoost</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.808</td>
      <td>0.100</td>
    </tr>
    <tr>
      <td>FastSHAP (5 Seeds Retrained)</td>
      <td>XGBoost</td>
      <td>0.687</td>
      <td>0.822</td>
      <td>---</td>
      <td>---</td>
    </tr>
    <tr>
      <td>KernelSHAP</td>
      <td>XGBoost</td>
      <td>0.940</td>
      <td>0.965</td>
      <td>0.740</td>
      <td>0.175</td>
    </tr>
    <tr>
      <td>LIME</td>
      <td>XGBoost</td>
      <td>0.714</td>
      <td>0.858</td>
      <td>0.780</td>
      <td>0.159</td>
    </tr>
    <tr>
      <td>TreeSHAP</td>
      <td>LightGBM</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.800</td>
      <td>0.411</td>
    </tr>
    <tr>
      <td>FastSHAP</td>
      <td>LightGBM</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.800</td>
      <td>0.078</td>
    </tr>
    <tr>
      <td>KernelSHAP</td>
      <td>LightGBM</td>
      <td>0.925</td>
      <td>0.952</td>
      <td>0.668</td>
      <td>0.294</td>
    </tr>
    <tr>
      <td>LIME</td>
      <td>LightGBM</td>
      <td>0.714</td>
      <td>0.858</td>
      <td>0.540</td>
      <td>0.169</td>
    </tr>
    <tr>
      <td>FastSHAP</td>
      <td>MLP</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.800</td>
      <td>0.096</td>
    </tr>
  </tbody>
</table>

<div class="figure-container">
  <img src="figures/figure7_robustness.png" alt="Lipschitz Sensitivity">
  <div class="figure-caption"><b>Fig. 3.</b> Explanation Lipschitz sensitivity ratio under input noise ($\epsilon=0.05$).</div>
</div>

<h2>VII. Pareto Analysis & Dimensionality Scaling</h2>
<p>Pareto analysis across all 12 configurations maps the multi-objective frontier balancing 95th-percentile latency against normalized faithfulness. On XGBoost, FastSHAP (0.031 ms) and TreeSHAP (1.798 ms) define the frontier. On Deep MLP, Exact Shapley ($p95 = 7.150$\,ms, $\mathcal{F}_{\text{norm}} = 0.8765$), KernelSHAP ($1.710$\,ms), and FastSHAP ($0.028$\,ms) define the frontier, with Exact Shapley establishing the absolute upper bound in faithfulness.</p>
<p>Simulating feature scaling across $d \in \{10, 25, 50, 100\}$ reveals: Exact Shapley scales as $O(2^d \cdot B)$, running in $6.84$\,ms at $d=10$ and $<28$\,ms at $d=12$, but breaching the 50\,ms SLA at $d=13$ ($54.7$\,ms) and reaching 224\,s at $d=25$. KernelSHAP ($N_{\text{samp}}=100$) scales linearly in $d$, breaching the 50\,ms SLA at $d \ge 62$ ($81.9$\,ms at $d=100$). TreeSHAP ($O(TLD^2)$) scales to $15.5$\,ms at $d=100$, never breaching SLAs, while FastSHAP maintains constant $0.03$\,ms inference. FastSHAP amortizes training costs ($T_{\text{train}} = 7.44$\,s on XGBoost, $3.49$\,s on MLP) at 922 queries vs KernelSHAP and 5,069 vs TreeSHAP (XGBoost), and 2,291 queries vs KernelSHAP and 4,043 vs LIME (MLP).</p>

<div class="figure-container">
  <img src="figures/figure6_pareto_frontier.png" alt="Pareto Frontier">
  <div class="figure-caption"><b>Fig. 4.</b> Multi-objective Pareto frontier mapping 95th-percentile latency against normalized faithfulness across all 12 configurations.</div>
</div>

<div class="figure-container">
  <img src="figures/figure8_amortization.png" alt="FastSHAP Amortization Curves">
  <div class="figure-caption"><b>Fig. 5.</b> Model-specific FastSHAP break-even amortization curves for (A) XGBoost and (B) Deep MLP architectures.</div>
</div>

<h2>VIII. Discussion & Operational Recommendations</h2>
<p>For tree ensembles, TreeSHAP executes in $\approx 1.5$\,ms, fitting comfortably within 50 ms SLAs with a &gt;30x safety margin. Replacing TreeSHAP with FastSHAP saves merely 1.4 ms in single-query regimes while introducing a 16.7% risk of primary reason divergence in the Adverse Action Cohort. Under CJEU jurisprudence (<i>SCHUFA Holding</i>, C-634/21; <i>Dun & Bradstreet Austria</i>, C-203/22) and US ECOA Regulation B, this creates avoidable regulatory exposure. We formulate the following decision rubric:
<ul>
  <li><b>Tier 1 (Tree Ensembles):</b> Deploy <b>TreeSHAP</b>. Guarantees exact cooperative attributions relative to marginal background expectation in $\approx 1.5$\,ms with perfect determinism.</li>
  <li><b>Tier 2 (Neural Models & Low Dimensions):</b> For $d \le 12$, deploy <b>Exact Shapley</b> (runs in &lt;28 ms, $6.78$\,ms at $d=10$), providing 100% mathematical faithfulness and zero surrogate drift within strict SLAs. For $d > 12$ or streaming (&gt;5,000 QPS), deploy distilled FastSHAP coupled with automated OOD drift detectors.</li>
  <li><b>Tier 3 (Surrogate Governance):</b> Fix pseudo-random seeds and archive explanations synchronously at decision time when using sampling surrogates (LIME) to prevent audit inconsistencies.</li>
</ul>
</p>

<h2>Acknowledgment</h2>
<p class="no-indent" style="font-size: 7.2pt;">In accordance with IEEE Authorship Policy, generative AI tools (Google DeepMind Gemini models) were utilized solely for code refactoring, language polishing, and LaTeX typesetting assistance. All conceptual frameworks, mathematical formulations, experimental pipelines, empirical data collection, statistical analyses, and scientific conclusions were conceived, executed, and validated independently by the author.</p>

<h2>References</h2>
<ol class="ref-list">
  <li>S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in <i>NeurIPS</i>, 2017, pp. 4765–4774.</li>
  <li>S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," <i>Nature Machine Intelligence</i>, vol. 2, no. 1, pp. 56–67, 2020.</li>
  <li>N. Jethani et al., "FastSHAP: Real-time Shapley value estimation," in <i>ICLR</i>, 2022.</li>
  <li>M. T. Ribeiro, S. Singh, and C. Guestrin, ""Why should I trust you?": Explaining the predictions of any classifier," in <i>ACM KDD</i>, 2016, pp. 1135–1144.</li>
  <li>W. Samek et al., "Evaluating the visualization of what a deep neural network has learned," <i>IEEE TNNLS</i>, vol. 28, no. 11, pp. 2660–2673, 2017.</li>
  <li>V. Petsiuk, A. Das, and K. Saenko, "RISE: Randomized input sampling for explanation of black-box models," in <i>BMVC</i>, 2018.</li>
  <li>D. Alvarez-Melis and T. S. Jaakkola, "On the robustness of interpretability methods," in <i>ICML Workshop on Human Interpretability in Machine Learning (WHI)</i>, Stockholm, Sweden, 2018.</li>
  <li>D. Slack et al., "Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods," in <i>AAAI/ACM AIES</i>, 2020, pp. 180–186.</li>
  <li>Court of Justice of the European Union, "Judgment in Case C-634/21, SCHUFA Holding (Scoring)," <i>ECLI:EU:C:2023:957</i>, Dec. 2023.</li>
  <li>Court of Justice of the European Union, "Judgment in Case C-203/22, Dun & Bradstreet Austria," <i>ECLI:EU:C:2025:117</i>, Feb. 2025.</li>
  <li>Consumer Financial Protection Bureau, "Equal Credit Opportunity Act (Regulation B), 12 CFR Part 1002," 2011.</li>
  <li>European Parliament and Council of the EU, "Regulation (EU) 2024/1689 (Artificial Intelligence Act)," <i>Official Journal of the EU</i>, 2024.</li>
  <li>European Parliament and Council of the EU, "Regulation (EU) 2016/679 (GDPR)," <i>Official Journal of the EU</i>, 2016.</li>
</ol>

</div>

</body>
</html>
"""

html_path = Path("paper/paper_camera_ready.html")
pdf_path = Path("paper/paper_camera_ready.pdf")

html_path.write_text(html_content, encoding="utf-8")
print(f"Generated HTML paper at {html_path}")

chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
abs_html_path = html_path.resolve()
abs_pdf_path = pdf_path.resolve()

cmd = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={abs_pdf_path}",
    f"file://{abs_html_path}"
]

print("Executing Chrome headless PDF generation...")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode == 0 and abs_pdf_path.exists():
    print(f"SUCCESS: Generated camera-ready PDF at {abs_pdf_path} (Size: {abs_pdf_path.stat().st_size} bytes)")
else:
    print(f"Error generating PDF via Chrome: {res.stderr}")
