from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from credit_xai.utils import get_logger, ensure_dirs

logger = get_logger("credit_xai.plotting")

# Configure publication-grade styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def plot_predictive_performance(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Path
) -> None:
    """Figure 2: ROC and Precision-Recall Curves for Base Prediction Models."""
    ensure_dirs(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ROC Curves
    ax_roc = axes[0]
    for model_name, (fpr, tpr, auc_val) in roc_data.items():
        ax_roc.plot(fpr, tpr, lw=1.8, label=f"{model_name} (AUC = {auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Random Guess")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    # PR Curves
    ax_pr = axes[1]
    for model_name, (rec, prec, pr_auc) in pr_data.items():
        ax_pr.plot(rec, prec, lw=1.8, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    ax_pr.set_title("Precision-Recall (PR) Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved predictive performance curves to {output_path}")

def plot_latency_distributions(raw_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 3: Latency Distributions across Explainers (Log-scale Violin & Box Plot)."""
    ensure_dirs(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    plot_df = raw_df[raw_df["success"] == True].copy()
    plot_df["explainer_label"] = plot_df["explainer_name"] + "\n(" + plot_df["model_name"] + ")"

    sns.boxplot(
        data=plot_df,
        x="explainer_label",
        y="xai_latency_ms",
        palette="Blues_r",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red"},
        ax=ax
    )
    ax.set_yscale("log")
    ax.set_title("Per-Instance Explanation Latency Distribution")
    ax.set_xlabel("Explanation Method (Model)")
    ax.set_ylabel("Latency (ms, Log Scale)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8.5)
    
    # SLA threshold lines
    ax.axhline(50.0, color="orange", linestyle="--", lw=1.2, label="Strict SLA (50 ms)")
    ax.axhline(100.0, color="red", linestyle="--", lw=1.2, label="Broad SLA (100 ms)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved latency distributions to {output_path}")

def plot_tail_latency(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 4: Tail Latencies (p50, p95, p99) against Service Level Agreements."""
    ensure_dirs(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5.0))

    x = np.arange(len(summary_df))
    width = 0.25

    p50 = summary_df["xai_p50_ms"].values
    p95 = summary_df["xai_p95_ms"].values
    p99 = summary_df["xai_p99_ms"].values
    labels = summary_df["explainer_name"] + "\n[" + summary_df["model_name"] + "]"

    ax.bar(x - width, p50, width, label="p50 (Median)", color="#2b5c8f")
    ax.bar(x, p95, width, label="p95 Percentile", color="#4682b4")
    ax.bar(x + width, p99, width, label="p99 Percentile", color="#87ceeb")

    ax.axhline(50.0, color="darkorange", linestyle="--", lw=1.5, label="Strict SLA: 50 ms")
    ax.axhline(100.0, color="crimson", linestyle="--", lw=1.5, label="Broad SLA: 100 ms")

    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, Log Scale)")
    ax.set_title("Tail Latency Benchmark Across Percentiles")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved tail latency plot to {output_path}")

def plot_deletion_curves(
    raw_curves: Dict[str, np.ndarray],
    norm_curves: Dict[str, np.ndarray],
    output_path: Path
) -> None:
    """Figure 5: 2-Panel Feature Deletion Curves: Raw Trajectory vs Normalized Scale-Invariant Degradation."""
    ensure_dirs(output_path.parent)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    colors = {
        "TreeSHAP (XGBoost)": "#1f77b4",
        "FastSHAP (XGBoost)": "#2ca02c",
        "KernelSHAP (XGBoost)": "#ff7f0e",
        "LIME (XGBoost)": "#d62728",
        "TreeSHAP (LightGBM)": "#17becf",
        "FastSHAP (LightGBM)": "#bcbd22",
        "KernelSHAP (LightGBM)": "#8c564b",
        "LIME (LightGBM)": "#e377c2",
        "FastSHAP (MLP)": "#9467bd",
        "KernelSHAP (MLP)": "#7f7f7f",
        "LIME (MLP)": "#aa40fc"
    }

    # Left: Raw confidence trajectory
    for label, curve in raw_curves.items():
        x_steps = np.arange(len(curve))
        c = colors.get(label, None)
        ax1.plot(x_steps, curve, marker="o", lw=1.8, markersize=3.5, label=label, color=c)

    ax1.set_title("A. Raw Prediction Probability Trajectory", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Number of Removed Features ($k$)", fontsize=10)
    ax1.set_ylabel("Predicted Default Probability $f(x^{(k)})$", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Right: Scale-invariant normalized deletion curve
    for label, curve in norm_curves.items():
        x_steps = np.arange(len(curve))
        c = colors.get(label, None)
        ax2.plot(x_steps, curve, marker="s", lw=1.8, markersize=3.5, label=label, color=c)

    ax2.set_title("B. Normalized Scale-Invariant Deletion $\\tilde{C}(k)$", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Number of Removed Features ($k$)", fontsize=10)
    ax2.set_ylabel("Normalized Probability $\\tilde{C}(k) \\in [0, 1]$", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved 2-panel deletion curves to {output_path}")

def plot_pareto_frontier(
    configs: List[Any],
    output_path: Path
) -> None:
    """Figure 6: 2D Pareto Frontier: Latency (p95) vs Explanation Faithfulness."""
    ensure_dirs(output_path.parent)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    latencies = [c.latency_p95_ms for c in configs]
    faithfulness = [c.faithfulness_score for c in configs]
    is_pareto = [c.is_pareto_optimal for c in configs]
    names = [f"{c.explainer_name} ({c.model_name})" for c in configs]

    # Plot dominated points
    dom_x = [latencies[i] for i in range(len(configs)) if not is_pareto[i]]
    dom_y = [faithfulness[i] for i in range(len(configs)) if not is_pareto[i]]
    ax.scatter(dom_x, dom_y, color="#7f7f7f", alpha=0.7, s=75, label="Dominated Configuration", zorder=3)

    # Plot Pareto optimal points
    par_x = [latencies[i] for i in range(len(configs)) if is_pareto[i]]
    par_y = [faithfulness[i] for i in range(len(configs)) if is_pareto[i]]
    ax.scatter(par_x, par_y, color="#d62728", s=130, edgecolors="black", linewidths=1.2, zorder=5, label="Pareto Optimal Frontier")

    # Draw step frontier
    if len(par_x) > 1:
        sorted_indices = np.argsort(par_x)
        px_sorted = np.array(par_x)[sorted_indices]
        py_sorted = np.array(par_y)[sorted_indices]
        ax.step(px_sorted, py_sorted, where="post", color="#d62728", linestyle="-.", lw=1.6, alpha=0.8)

    # Explicit, non-overlapping annotation offsets for each configuration
    offsets_map = {
        "FastSHAP (MLP)": (12, -14),
        "FastSHAP (XGBoost)": (12, 14),
        "FastSHAP (LightGBM)": (12, -18),
        "TreeSHAP (XGBoost)": (18, -4),
        "TreeSHAP (LightGBM)": (14, -20),
        "ExactShapley (MLP)": (-118, 14),
        "KernelSHAP (MLP)": (-105, 18),
        "KernelSHAP (XGBoost)": (14, 8),
        "KernelSHAP (LightGBM)": (14, -20),
        "LIME (MLP)": (-65, -20),
        "LIME (XGBoost)": (-95, -16),
        "LIME (LightGBM)": (14, -22)
    }

    # Label points with collision prevention
    for i, name in enumerate(names):
        x_pt, y_pt = latencies[i], faithfulness[i]
        offset_x, offset_y = offsets_map.get(name, (10, 10))

        ax.annotate(
            name,
            (x_pt, y_pt),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=8.5,
            fontweight="bold" if is_pareto[i] else "normal",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#cccccc", lw=0.6)
        )

    ax.set_xscale("log")
    ax.set_ylim(0.55, 0.92)
    ax.axvline(50.0, color="orange", linestyle="--", lw=1.2, label="Strict SLA (50 ms)")
    ax.axvline(100.0, color="red", linestyle="--", lw=1.2, label="Broad SLA (100 ms)")

    ax.set_title("Pareto Frontier: Explanation Latency vs Normalized Faithfulness", fontsize=11, fontweight="bold", pad=14)
    ax.set_xlabel("95th Percentile Latency (ms, Log Scale - Lower is Better)", fontsize=10)
    ax.set_ylabel("Normalized Faithfulness Score $\\mathcal{F}_{\\text{norm}}$ (Higher is Better)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    # Place legend in lower right to avoid collision with low-latency points
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    logger.info(f"Saved Pareto frontier plot to {output_path}")

def plot_fastshap_amortization(
    amort_dict: Dict[str, Any],
    max_queries: int = 6500,
    output_path: Path = Path("results/figures/figure8_amortization.png")
) -> None:
    """Figure 8: Model-Specific FastSHAP Break-Even Amortization Analysis."""
    ensure_dirs(output_path.parent)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0))

    queries = np.linspace(1, max_queries, 650)

    # Panel 1: Tree Ensembles (XGBoost)
    if "xgb" in amort_dict:
        data_xgb = amort_dict["xgb"]
        t_train = data_xgb["fastshap_training_seconds"]
        l_fast = data_xgb["fastshap_latency_ms"] / 1000.0
        cost_fast = t_train + queries * l_fast

        ax1.plot(queries, cost_fast, label=f"FastSHAP (Train: {t_train:.2f}s + {data_xgb['fastshap_latency_ms']:.3f}ms/q)", color="#2ca02c", lw=2)

        # vs TreeSHAP
        if "vs_treeshap" in data_xgb:
            l_tree = data_xgb["vs_treeshap"]["latency_ms"] / 1000.0
            q_star = data_xgb["vs_treeshap"]["break_even_queries"]
            ax1.plot(queries, queries * l_tree, label=f"TreeSHAP ({data_xgb['vs_treeshap']['latency_ms']:.2f}ms/q)", color="#1f77b4", lw=2, linestyle="--")
            if q_star <= max_queries:
                ax1.plot(q_star, q_star * l_tree, "ro", markersize=7)
                ax1.annotate(
                    f"Break-Even vs TreeSHAP: {q_star} q",
                    (q_star, q_star * l_tree),
                    textcoords="offset points",
                    xytext=(-155, 16),
                    fontsize=8.5,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#1f77b4", lw=0.8)
                )

        # vs KernelSHAP
        if "vs_kernelshap" in data_xgb:
            l_kern = data_xgb["vs_kernelshap"]["latency_ms"] / 1000.0
            q_star_k = data_xgb["vs_kernelshap"]["break_even_queries"]
            ax1.plot(queries, queries * l_kern, label=f"KernelSHAP ({data_xgb['vs_kernelshap']['latency_ms']:.2f}ms/q)", color="#ff7f0e", lw=1.8, linestyle=":")
            if q_star_k <= max_queries:
                ax1.plot(q_star_k, q_star_k * l_kern, "ks", markersize=6)
                ax1.annotate(
                    f"Break-Even vs Kernel: {q_star_k} q",
                    (q_star_k, q_star_k * l_kern),
                    textcoords="offset points",
                    xytext=(15, -20),
                    fontsize=8.5,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#ff7f0e", lw=0.8)
                )

        ax1.set_xlim(0, max_queries)
        ax1.set_title("A. XGBoost Amortization Economics", fontsize=11, fontweight="bold", pad=12)
        ax1.set_xlabel("Production Query Volume ($Q$)", fontsize=10)
        ax1.set_ylabel("Cumulative Compute Time (Seconds)", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", fontsize=8.5)

    # Panel 2: Neural Architecture (PyTorch MLP)
    if "mlp" in amort_dict:
        data_mlp = amort_dict["mlp"]
        t_train_m = data_mlp["fastshap_training_seconds"]
        l_fast_m = data_mlp["fastshap_latency_ms"] / 1000.0
        cost_fast_m = t_train_m + queries * l_fast_m

        ax2.plot(queries, cost_fast_m, label=f"FastSHAP (Train: {t_train_m:.2f}s + {data_mlp['fastshap_latency_ms']:.3f}ms/q)", color="#9467bd", lw=2)

        # vs KernelSHAP
        if "vs_kernelshap" in data_mlp:
            l_kern_m = data_mlp["vs_kernelshap"]["latency_ms"] / 1000.0
            q_star_km = data_mlp["vs_kernelshap"]["break_even_queries"]
            ax2.plot(queries, queries * l_kern_m, label=f"KernelSHAP ({data_mlp['vs_kernelshap']['latency_ms']:.2f}ms/q)", color="#ff7f0e", lw=2, linestyle="--")
            if q_star_km <= max_queries:
                ax2.plot(q_star_km, q_star_km * l_kern_m, "ro", markersize=7)
                ax2.annotate(
                    f"Break-Even vs Kernel: {q_star_km} q",
                    (q_star_km, q_star_km * l_kern_m),
                    textcoords="offset points",
                    xytext=(15, 16),
                    fontsize=8.5,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#ff7f0e", lw=0.8)
                )

        # vs LIME
        if "vs_lime" in data_mlp:
            l_lime_m = data_mlp["vs_lime"]["latency_ms"] / 1000.0
            q_star_lm = data_mlp["vs_lime"]["break_even_queries"]
            ax2.plot(queries, queries * l_lime_m, label=f"LIME ({data_mlp['vs_lime']['latency_ms']:.2f}ms/q)", color="#d62728", lw=1.8, linestyle=":")
            if q_star_lm <= max_queries:
                ax2.plot(q_star_lm, q_star_lm * l_lime_m, "ks", markersize=6)
                ax2.annotate(
                    f"Break-Even vs LIME: {q_star_lm} q",
                    (q_star_lm, q_star_lm * l_lime_m),
                    textcoords="offset points",
                    xytext=(15, -22),
                    fontsize=8.5,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#d62728", lw=0.8)
                )

        ax2.set_xlim(0, max_queries)
        ax2.set_title("B. Deep MLP Amortization Economics", fontsize=11, fontweight="bold", pad=12)
        ax2.set_xlabel("Production Query Volume ($Q$)", fontsize=10)
        ax2.set_ylabel("Cumulative Compute Time (Seconds)", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout(pad=2.2)
    fig.subplots_adjust(top=0.90)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close()
    logger.info(f"Saved model-specific amortization plot to {output_path}")
