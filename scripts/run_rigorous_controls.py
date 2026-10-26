import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from credit_xai.config import ExperimentConfig
from credit_xai.data import FEATURE_NAMES
from credit_xai.preprocessing import CreditDataPreprocessor
from credit_xai.explainers import (
    TreeSHAPExplainer,
    KernelSHAPExplainer,
    FastSHAPExplainer,
    LIMEExplainer,
    ExactShapleyExplainer
)
from credit_xai.faithfulness import FaithfulnessEvaluator
from credit_xai.utils import set_seeds

def main():
    set_seeds(42)
    cfg = ExperimentConfig.from_yaml("configs/experiment.yaml")

    train_df = pd.read_parquet(Path(cfg.data.processed_dir) / cfg.data.processed_train_filename)
    test_df = pd.read_parquet(Path(cfg.data.processed_dir) / cfg.data.processed_test_filename)

    preprocessor = CreditDataPreprocessor()
    preprocessor.fit(train_df)

    X_train, y_train, _ = preprocessor.transform(train_df)
    X_test, y_test, _ = preprocessor.transform(test_df)

    model_dir = Path("results/metadata/models")
    with open(model_dir / "xgboost.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(model_dir / "lightgbm.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    with open(model_dir / "mlp.pkl", "rb") as f:
        mlp_model = pickle.load(f)

    X_bg = X_train[:100]

    print("================================================================")
    print("1. FULL TEST SET (N=22,500) EVALUATION: TreeSHAP vs FastSHAP")
    print("================================================================")
    tree_xgb = TreeSHAPExplainer(xgb_model, background_data=X_bg, model_output="probability")
    
    # Train FastSHAP with TreeSHAP distillation on 3,000 samples
    t0_teach = time.perf_counter()
    Y_train_tree = tree_xgb.explain_batch(X_train[:3000])
    t_teach = time.perf_counter() - t0_teach
    print(f"Precomputed 3,000 TreeSHAP teacher attributions in {t_teach:.2f} s")

    fast_xgb = FastSHAPExplainer(xgb_model.predict_proba, background_data=X_bg, epochs=100, batch_size=64, lr=0.003)
    fast_xgb.train_explainer(X_train[:3000], teacher_attributions=Y_train_tree)

    print("Explaining full test set (N=22,500)...")
    t0_tree_full = time.perf_counter()
    phi_tree_full = tree_xgb.explain_batch(X_test)
    t_tree_full = time.perf_counter() - t0_tree_full
    print(f"TreeSHAP full test set (22,500): {t_tree_full:.2f} s ({t_tree_full/len(X_test)*1000:.3f} ms/sample)")

    t0_fast_full = time.perf_counter()
    phi_fast_full = fast_xgb.explain_batch(X_test)
    t_fast_full = time.perf_counter() - t0_fast_full
    print(f"FastSHAP full test set (22,500): {t_fast_full:.2f} s ({t_fast_full/len(X_test)*1000:.3f} ms/sample)")

    # Overall cohort (N=22,500)
    top1_tree = np.argmax(phi_tree_full, axis=1)
    top1_fast = np.argmax(phi_fast_full, axis=1)
    top1_match_overall = np.mean(top1_tree == top1_fast)
    
    # Wilson score interval for top1
    n = len(X_test)
    p = top1_match_overall
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    ci_overall = (center - half, center + half)

    # Top-3 and Top-4 Jaccard
    jaccards_3 = []
    jaccards_4 = []
    rhos = []
    for i in range(len(X_test)):
        t_order = np.argsort(phi_tree_full[i])[::-1]
        f_order = np.argsort(phi_fast_full[i])[::-1]
        j3 = len(set(t_order[:3]) & set(f_order[:3])) / len(set(t_order[:3]) | set(f_order[:3]))
        j4 = len(set(t_order[:4]) & set(f_order[:4])) / len(set(t_order[:4]) | set(f_order[:4]))
        jaccards_3.append(j3)
        jaccards_4.append(j4)
        r, _ = spearmanr(phi_tree_full[i], phi_fast_full[i])
        if not np.isnan(r):
            rhos.append(r)

    print(f"\nOVERALL COHORT (N={len(X_test)}):")
    print(f"  Top-1 Agreement: {top1_match_overall*100:.2f}% (95% CI: [{ci_overall[0]*100:.2f}%, {ci_overall[1]*100:.2f}%])")
    print(f"  Top-3 Jaccard Similarity: {np.mean(jaccards_3)*100:.2f}%")
    print(f"  Top-4 Jaccard Similarity: {np.mean(jaccards_4)*100:.2f}%")
    print(f"  Spearman rho_s: {np.mean(rhos):.4f}")
    print(f"  Attribution MSE: {np.mean((phi_tree_full - phi_fast_full)**2):.6f}")

    # Adverse Action Cohort: applicants with f(x) >= 0.50 or top risk decile
    preds_xgb = xgb_model.predict_proba(X_test)[:, 1]
    adverse_mask_prob = preds_xgb >= 0.50
    adverse_mask_decile = preds_xgb >= np.percentile(preds_xgb, 90)
    adverse_mask = adverse_mask_prob | adverse_mask_decile
    print(f"\nADVERSE ACTION COHORT (N={np.sum(adverse_mask)}):")
    p_adv = np.mean(top1_tree[adverse_mask] == top1_fast[adverse_mask])
    n_adv = np.sum(adverse_mask)
    denom_adv = 1 + z**2 / n_adv
    center_adv = (p_adv + z**2 / (2 * n_adv)) / denom_adv
    half_adv = z * np.sqrt(p_adv * (1 - p_adv) / n_adv + z**2 / (4 * n_adv**2)) / denom_adv
    ci_adv = (center_adv - half_adv, center_adv + half_adv)
    print(f"  Top-1 Agreement: {p_adv*100:.2f}% (95% CI: [{ci_adv[0]*100:.2f}%, {ci_adv[1]*100:.2f}%])")
    print(f"  Top-3 Jaccard Similarity: {np.mean(np.array(jaccards_3)[adverse_mask])*100:.2f}%")
    print(f"  Top-4 Jaccard Similarity: {np.mean(np.array(jaccards_4)[adverse_mask])*100:.2f}%")
    print(f"  Spearman rho_s: {np.mean(np.array(rhos)[adverse_mask]):.4f}")

    # Near-tie analysis: gap between top-1 and top-2
    sorted_tree = np.sort(phi_tree_full, axis=1)[:, ::-1]
    gaps = sorted_tree[:, 0] - sorted_tree[:, 1]
    clear_winner_mask = gaps > 0.01
    print(f"\nCLEAR WINNER COHORT (Gap > 0.01, N={np.sum(clear_winner_mask)} / {len(X_test)}):")
    p_clear = np.mean(top1_tree[clear_winner_mask] == top1_fast[clear_winner_mask])
    print(f"  Top-1 Agreement: {p_clear*100:.2f}%")

    print("\n================================================================")
    print("2. PAIRED BOOTSTRAP HYPOTHESIS TESTING FOR FAITHFULNESS (F_norm)")
    print("================================================================")
    eval_xgb = FaithfulnessEvaluator(xgb_model.predict_proba, background_data=X_bg)
    eval_mlp = FaithfulnessEvaluator(mlp_model.predict_proba, background_data=X_bg)

    X_eval = X_test[:200]
    # Compute per-instance normalized faithfulness for TreeSHAP, FastSHAP, KernelSHAP, LIME on XGBoost
    kernel_xgb = KernelSHAPExplainer(xgb_model.predict_proba, background_data=X_bg, nsamples=100)
    lime_xgb = LIMEExplainer(xgb_model.predict_proba, training_data=X_train[:1000], num_samples=500)

    # Subsets for paired bootstrap
    N_pair = 50
    X_pair = X_eval[:N_pair]

    f_tree = []
    f_fast = []
    f_kernel = []
    f_lime = []

    print(f"Computing per-instance faithfulness for {N_pair} instances across XGBoost explainers...")
    for i in range(N_pair):
        x_i = X_pair[i]
        attr_t = tree_xgb.explain_instance(x_i)
        attr_f = fast_xgb.explain_instance(x_i)
        attr_k = kernel_xgb.explain_instance(x_i)
        attr_l = lime_xgb.explain_instance(x_i)

        _, _, _, n_auc_t = eval_xgb.compute_deletion_curve(x_i, attr_t)
        _, _, _, n_auc_f = eval_xgb.compute_deletion_curve(x_i, attr_f)
        _, _, _, n_auc_k = eval_xgb.compute_deletion_curve(x_i, attr_k)
        _, _, _, n_auc_l = eval_xgb.compute_deletion_curve(x_i, attr_l)

        f_tree.append(1.0 - n_auc_t)
        f_fast.append(1.0 - n_auc_f)
        f_kernel.append(1.0 - n_auc_k)
        f_lime.append(1.0 - n_auc_l)

    f_tree = np.array(f_tree)
    f_fast = np.array(f_fast)
    f_kernel = np.array(f_kernel)
    f_lime = np.array(f_lime)

    def paired_bootstrap(arr_a, arr_b, B=2000, seed=42):
        rng = np.random.RandomState(seed)
        diff = arr_a - arr_b
        mean_diff = np.mean(diff)
        boot_diffs = []
        for _ in range(B):
            idx = rng.choice(len(diff), size=len(diff), replace=True)
            boot_diffs.append(np.mean(diff[idx]))
        boot_diffs = np.array(boot_diffs)
        ci = (np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5))
        # Two-sided empirical p-value for H0: mean_diff == 0
        p_val = 2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
        return mean_diff, ci, p_val

    pairs = [
        ("TreeSHAP", "FastSHAP", f_tree, f_fast),
        ("TreeSHAP", "KernelSHAP", f_tree, f_kernel),
        ("TreeSHAP", "LIME", f_tree, f_lime),
        ("FastSHAP", "KernelSHAP", f_fast, f_kernel),
        ("FastSHAP", "LIME", f_fast, f_lime),
        ("KernelSHAP", "LIME", f_kernel, f_lime)
    ]

    print("\nPaired Bootstrap Results (B=2,000 resamples):")
    for name_a, name_b, a, b in pairs:
        md, ci, p_val = paired_bootstrap(a, b)
        print(f"  {name_a} vs {name_b}:")
        print(f"    Delta F_norm: {md:+.4f} (95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]), p = {p_val:.4f}")

    print("\n================================================================")
    print("3. MULTI-METHOD OOD COVARIATE SHIFT (+3.0 sigma shock)")
    print("================================================================")
    # Apply +3.0 sigma shock to RevolvingUtilizationOfUnsecuredLines (idx 0) and DebtRatio (idx 4)
    X_ood = X_pair.copy()
    X_ood[:, 0] += 3.0
    X_ood[:, 4] += 3.0

    print("Evaluating attribution distortion under +3.0 sigma shift across TreeSHAP, KernelSHAP, LIME, FastSHAP...")
    ood_results = {}
    for name, explainer in [("TreeSHAP", tree_xgb), ("FastSHAP", fast_xgb), ("KernelSHAP", kernel_xgb), ("LIME", lime_xgb)]:
        phi_in = []
        phi_shift = []
        for x_i, x_o in zip(X_pair, X_ood):
            phi_in.append(explainer.explain_instance(x_i))
            phi_shift.append(explainer.explain_instance(x_o))
        phi_in = np.array(phi_in)
        phi_shift = np.array(phi_shift)

        raw_mse = np.mean((phi_shift - phi_in)**2)
        norm_mse = np.mean(np.sum((phi_shift - phi_in)**2, axis=1) / (np.sum(phi_in**2, axis=1) + 1e-8))
        
        rhos_shift = [spearmanr(phi_in[i], phi_shift[i])[0] for i in range(len(X_pair)) if not np.isnan(spearmanr(phi_in[i], phi_shift[i])[0])]
        mean_rho = np.mean(rhos_shift)

        # Bootstrap CI for NMSE
        rng = np.random.RandomState(42)
        nmse_boots = [np.mean(np.sum((phi_shift[idx] - phi_in[idx])**2, axis=1) / (np.sum(phi_in[idx]**2, axis=1) + 1e-8))
                      for idx in [rng.choice(len(X_pair), size=len(X_pair), replace=True) for _ in range(1000)]]
        ci_nmse = (np.percentile(nmse_boots, 2.5), np.percentile(nmse_boots, 97.5))

        ood_results[name] = {
            "raw_mse": raw_mse,
            "nmse": norm_mse,
            "nmse_ci": ci_nmse,
            "spearman_rho": mean_rho
        }
        print(f"  {name:10s} | Raw MSE: {raw_mse:.5f} | NMSE: {norm_mse:.4f} (95% CI: [{ci_nmse[0]:.4f}, {ci_nmse[1]:.4f}]) | rho_s: {mean_rho:.4f}")

    print("\n================================================================")
    print("4. HIGH-DIMENSIONAL SCALING SIMULATION (d in {10, 25, 50, 100})")
    print("================================================================")
    dimensions = [10, 25, 50, 100]
    for d in dimensions:
        exact_evals = 2**d
        exact_ms = (exact_evals / 1024) * 6.84
        # KernelSHAP evaluates nsamples coalitions
        kernel_500_ms = 8.19 * (d / 10.0) # linear in d for solving ridge regression
        # TreeSHAP scales O(L * D^2) roughly linear in d
        tree_ms = 1.55 * (d / 10.0)
        # FastSHAP is O(1) single neural forward pass
        fast_ms = 0.031

        print(f"Dimension d={d:3d}:")
        print(f"  Exact Shapley (MLP):  {exact_ms:12.2f} ms | SLA 50ms Breach? {'YES (BREACH)' if exact_ms > 50 else 'NO'}")
        print(f"  KernelSHAP (500):     {kernel_500_ms:12.2f} ms | SLA 50ms Breach? {'YES (BREACH)' if kernel_500_ms > 50 else 'NO'}")
        print(f"  TreeSHAP (XGBoost):   {tree_ms:12.2f} ms | SLA 50ms Breach? {'YES (BREACH)' if tree_ms > 50 else 'NO'}")
        print(f"  FastSHAP:             {fast_ms:12.2f} ms | SLA 50ms Breach? {'YES (BREACH)' if fast_ms > 50 else 'NO'}")

if __name__ == "__main__":
    main()
