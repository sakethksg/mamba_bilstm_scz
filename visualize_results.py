"""
Publication-Quality Visualization Suite for Mamba-BiLSTM Hybrid SCZ Classification.
Generates:
  1. Fig 1: Cross-Validation Performance Across Folds (Accuracy, Bal Acc, AUC, Sens, Spec, F1)
  2. Fig 2: Receiver Operating Characteristic (ROC) Curves per Fold & Pooled Cohort
  3. Fig 3: Confusion Matrices (Standard Threshold 0.50 vs Optimal Calibrated Threshold)
  4. Fig 4: Subject-Level Predicted Probability Distributions (HC vs AVH- vs AVH+)
  5. Fig 5: Training & Validation Convergence Trajectories Across Folds
  6. Fig 6: Altered Connectome Functional Connectivity (SCZ vs HC Differential Topology)
"""
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from config import MambaBiLSTMConfig
from dataset_sliding_window import load_participants_data

# Set publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def generate_all_figures(
    results_dir: Path = Path("/root/fMRI/mamba_bilstm_scz/results"),
    logs_dir: Path = Path("/root/fMRI/mamba_bilstm_scz/logs"),
    config: MambaBiLSTMConfig = None
):
    if config is None:
        config = MambaBiLSTMConfig()
        
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    cv_metrics_path = results_dir / "cv_metrics.json"
    with open(cv_metrics_path, "r") as f:
        data = json.load(f)
        
    fold_results = data["fold_results"]
    mean_metrics = data["mean_metrics"]
    oof_preds = data["oof_predictions"]
    
    meta = load_participants_data(config.participants_tsv)
    sub_meta = {m["participant_id"]: m for m in meta}
    
    y_true_all = []
    y_prob_all = []
    groups_all = []
    subs_all = []
    for sub_id, prob in oof_preds.items():
        y_true_all.append(sub_meta[sub_id]["binary_label"])
        y_prob_all.append(prob)
        groups_all.append(sub_meta[sub_id]["group"])
        subs_all.append(sub_id)
        
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    
    print(f"Loaded {len(y_true_all)} subjects for visualization.")

    # =========================================================================
    # FIGURE 1: Cross-Validation Performance Comparison Across Folds
    # =========================================================================
    fig, ax = plt.subplots(figsize=(11, 5.5))
    metrics_names = ["accuracy", "balanced_accuracy", "roc_auc", "recall_sensitivity", "specificity", "f1_score"]
    display_names = ["Accuracy", "Balanced Acc", "ROC-AUC", "Sensitivity", "Specificity", "F1-Score"]
    
    n_folds = len(fold_results)
    x = np.arange(len(metrics_names))
    width = 0.13
    
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    for i, f_res in enumerate(fold_results):
        vals = [f_res[m] for m in metrics_names]
        ax.bar(x + (i - 2.5) * width, vals, width, label=f"Fold {i+1}", color=colors[i], alpha=0.85, edgecolor="black", linewidth=0.6)
        
    # Mean bar
    means = [mean_metrics[m] if m != "recall_sensitivity" else mean_metrics["sensitivity"] for m in metrics_names]
    stds = [mean_metrics[f"{m}_std"] if m != "recall_sensitivity" else mean_metrics["sensitivity_std"] for m in metrics_names]
    ax.bar(x + 2.5 * width, means, width, yerr=stds, capsize=4, label="Mean ± SD", color="#333333", alpha=0.9, edgecolor="black", linewidth=0.8)
    
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Mamba-BiLSTM Hybrid 5-Fold Stratified Cross-Validation (N=71 Cohort)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="Chance Level (0.50)")
    ax.legend(loc="upper right", ncol=4, frameon=True, framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    fig1_path = fig_dir / "fig1_cv_performance_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.savefig(fig_dir / "fig1_cv_performance_comparison.pdf")
    plt.close()
    print(f"Saved: {fig1_path}")

    # =========================================================================
    # FIGURE 2: ROC Curves (Per-Fold & Pooled Out-of-Fold)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    
    # Per fold ROC
    fold_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=len(fold_results), shuffle=True, random_state=config.seed)
    subs_order = [m["participant_id"] for m in meta]
    labels_order = [m["binary_label"] for m in meta]
    for i, (_, val_idx) in enumerate(skf.split(subs_order, labels_order)):
        f_subs = [subs_order[j] for j in val_idx]
        f_y = [labels_order[j] for j in val_idx]
        f_probs = [oof_preds[s] for s in f_subs]
        f_fpr, f_tpr, _ = roc_curve(f_y, f_probs)
        f_auc = fold_results[i]["roc_auc"]
        ax.plot(f_fpr, f_tpr, color=fold_colors[i], label=f"Fold {i+1} (AUC = {f_auc:.3f})", linewidth=1.6, alpha=0.75)
        
    # Pooled OOF ROC
    fpr, tpr, thresholds = roc_curve(y_true_all, y_prob_all)
    oof_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#D95F02", linewidth=3.0, label=f"Pooled OOF Cohort (AUC = {oof_auc:.3f})")
    
    # Youden J optimal point
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    ax.scatter(fpr[best_idx], tpr[best_idx], s=120, color="#E7298A", zorder=5, 
               edgecolor="black", label=f"Optimal Operating Point (Thresh = {best_thresh:.2f})")
    
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="Chance Level (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Validation", fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    fig2_path = fig_dir / "fig2_roc_curves.png"
    plt.savefig(fig2_path, dpi=300)
    plt.savefig(fig_dir / "fig2_roc_curves.pdf")
    plt.close()
    print(f"Saved: {fig2_path}")

    # =========================================================================
    # FIGURE 3: Confusion Matrices (Standard 0.50 vs Optimal Calibrated)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    
    # 1. Standard Threshold 0.50
    y_pred_std = (y_prob_all >= 0.50).astype(int)
    cm_std = confusion_matrix(y_true_all, y_pred_std, labels=[0, 1])
    
    sns.heatmap(cm_std, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                annot_kws={"size": 18, "weight": "bold"}, square=True,
                xticklabels=["HC (Pred)", "SCZ (Pred)"], yticklabels=["HC (True)", "SCZ (True)"])
    
    acc_std = np.trace(cm_std) / np.sum(cm_std)
    sens_std = cm_std[1, 1] / np.sum(cm_std[1, :])
    spec_std = cm_std[0, 0] / np.sum(cm_std[0, :])
    axes[0].set_title(f"Standard Decision Boundary (Threshold = 0.50)\nAccuracy: {acc_std*100:.1f}% | Sens: {sens_std*100:.1f}% | Spec: {spec_std*100:.1f}%", fontweight="bold", pad=10)
    
    # 2. Optimal Calibrated Threshold
    opt_t = float(data["oof_metrics"].get("optimal_threshold", 0.52))
    y_pred_opt = (y_prob_all >= opt_t).astype(int)
    cm_opt = confusion_matrix(y_true_all, y_pred_opt, labels=[0, 1])
    
    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1],
                annot_kws={"size": 18, "weight": "bold"}, square=True,
                xticklabels=["HC (Pred)", "SCZ (Pred)"], yticklabels=["HC (True)", "SCZ (True)"])
    
    acc_opt = np.trace(cm_opt) / np.sum(cm_opt)
    sens_opt = cm_opt[1, 1] / np.sum(cm_opt[1, :])
    spec_opt = cm_opt[0, 0] / np.sum(cm_opt[0, :])
    axes[1].set_title(f"Optimal Calibrated Decision Boundary (Threshold = {opt_t:.2f})\nAccuracy: {acc_opt*100:.1f}% | Sens: {sens_opt*100:.1f}% | Spec: {spec_opt*100:.1f}%", fontweight="bold", pad=10)
    
    plt.tight_layout()
    fig3_path = fig_dir / "fig3_confusion_matrices.png"
    plt.savefig(fig3_path, dpi=300)
    plt.savefig(fig_dir / "fig3_confusion_matrices.pdf")
    plt.close()
    print(f"Saved: {fig3_path}")

    # =========================================================================
    # FIGURE 4: Subject Predicted Diagnostic Probabilities (HC vs AVH- vs AVH+)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    subgroups = ["HC", "AVH-", "AVH+"]
    group_colors = {"HC": "#2B83BA", "AVH-": "#FDAE61", "AVH+": "#D7191C"}
    
    jitter_data = []
    labels_list = []
    
    for grp in subgroups:
        vals = [p for s, p, g in zip(subs_all, y_prob_all, groups_all) if g == grp]
        jitter_data.append(vals)
        labels_list.append(f"{grp}\n(n={len(vals)})")
        
    parts = ax.violinplot(jitter_data, positions=[1, 2, 3], showmeans=True, showmedians=False, showextrema=False, widths=0.6)
    for pc, grp in zip(parts["bodies"], subgroups):
        pc.set_facecolor(group_colors[grp])
        pc.set_edgecolor("black")
        pc.set_alpha(0.4)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(2)
    
    # Scatter points with slight jitter
    np.random.seed(42)
    for pos, (grp, vals) in enumerate(zip(subgroups, jitter_data), 1):
        x_jitter = np.random.normal(pos, 0.05, size=len(vals))
        ax.scatter(x_jitter, vals, color=group_colors[grp], s=55, alpha=0.9, edgecolor="black", linewidth=0.7, label=f"{grp} Participants" if pos==1 else None)
        
    ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.5, label="Standard Threshold (0.50)")
    ax.axhline(opt_t, color="#E7298A", linestyle=":", linewidth=2.0, label=f"Calibrated Threshold ({opt_t:.2f})")
    
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels_list, fontweight="bold")
    ax.set_ylabel("Predicted Probability of Schizophrenia P(SCZ)", fontweight="bold")
    ax.set_title("Subject-Level Diagnostic Posterior Separation Across Phenotypic Cohorts", fontweight="bold", pad=12)
    ax.set_ylim([-0.05, 1.05])
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    
    plt.tight_layout()
    fig4_path = fig_dir / "fig4_subject_diagnostic_probabilities.png"
    plt.savefig(fig4_path, dpi=300)
    plt.savefig(fig_dir / "fig4_subject_diagnostic_probabilities.pdf")
    plt.close()
    print(f"Saved: {fig4_path}")

    # =========================================================================
    # FIGURE 5: Training and Validation Convergence Across Folds
    # =========================================================================
    log_file = logs_dir / "mamba_bilstm_training_20261022_235144.log"
    fold_train_losses = {}
    fold_val_accs = {}
    current_fold = None
    
    if log_file.exists():
        with open(log_file, "r") as f:
            for line in f:
                fold_match = re.search(r">>> \[Fold (\d)/5\]", line)
                if fold_match:
                    current_fold = int(fold_match.group(1))
                    fold_train_losses[current_fold] = []
                    fold_val_accs[current_fold] = []
                    
                ep_match = re.search(r"\[Epoch (\d+)/075\] Loss: ([\d\.]+) \| WinAcc: [\d\.]+ \| LR: [\d\.\-e]+ \| Val Acc: ([\d\.]+)", line)
                if ep_match and current_fold is not None:
                    ep = int(ep_match.group(1))
                    loss = float(ep_match.group(2))
                    v_acc = float(ep_match.group(3))
                    fold_train_losses[current_fold].append((ep, loss))
                    fold_val_accs[current_fold].append((ep, v_acc))
                    
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for fid in sorted(fold_train_losses.keys()):
            pts = fold_train_losses[fid]
            if pts:
                eps, losses = zip(*pts)
                axes[0].plot(eps, losses, label=f"Fold {fid}", color=colors[fid-1], linewidth=1.8)
                
        axes[0].set_xlabel("Epoch", fontweight="bold")
        axes[0].set_ylabel("Focal Cross-Entropy Loss", fontweight="bold")
        axes[0].set_title("Training Loss Convergence (BF16 Autocast)", fontweight="bold")
        axes[0].grid(True, linestyle=":", alpha=0.5)
        axes[0].legend(loc="upper right", frameon=True)
        
        for fid in sorted(fold_val_accs.keys()):
            pts = fold_val_accs[fid]
            if pts:
                eps, vaccs = zip(*pts)
                axes[1].plot(eps, vaccs, label=f"Fold {fid}", color=colors[fid-1], linewidth=1.8)
                
        axes[1].set_xlabel("Epoch", fontweight="bold")
        axes[1].set_ylabel("Validation Subject Accuracy", fontweight="bold")
        axes[1].set_title("Validation Accuracy Trajectories", fontweight="bold")
        axes[1].set_ylim([0.2, 1.05])
        axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        axes[1].grid(True, linestyle=":", alpha=0.5)
        axes[1].legend(loc="lower right", frameon=True)
        
        plt.tight_layout()
        fig5_path = fig_dir / "fig5_training_convergence.png"
        plt.savefig(fig5_path, dpi=300)
        plt.savefig(fig_dir / "fig5_training_convergence.pdf")
        plt.close()
        print(f"Saved: {fig5_path}")

    # =========================================================================
    # FIGURE 6: Altered Connectome Functional Connectivity (SCZ vs HC Differential)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    hc_fcs = []
    scz_fcs = []
    
    for sub_id, m in sub_meta.items():
        p = config.roi_dir / f"{sub_id}_roi_timeseries.npy"
        if p.exists():
            ts = np.load(p) # (200, 341)
            ts_norm = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)
            corr = np.dot(ts_norm, ts_norm.T) / (ts.shape[1] - 1)
            np.fill_diagonal(corr, 0.0)
            corr_z = np.arctanh(np.clip(corr, -0.999, 0.999))
            if m["binary_label"] == 0:
                hc_fcs.append(corr_z)
            else:
                scz_fcs.append(corr_z)
                
    if hc_fcs and scz_fcs:
        mean_hc_fc = np.mean(hc_fcs, axis=0)
        mean_scz_fc = np.mean(scz_fcs, axis=0)
        diff_fc = mean_scz_fc - mean_hc_fc # SCZ - HC
        
        vmax = np.percentile(np.abs(diff_fc), 98)
        im = axes[0].imshow(diff_fc, cmap="vlag", vmin=-vmax, vmax=vmax, aspect="auto")
        axes[0].set_title(r"Differential Connectome ($\Delta FC = FC_{SCZ} - FC_{HC}$)" + "\nSchaefer 200 Parcels", fontweight="bold", pad=10)
        axes[0].set_xlabel("Cortical Parcels (1 to 200)", fontweight="bold")
        axes[0].set_ylabel("Cortical Parcels (1 to 200)", fontweight="bold")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label=r"$\Delta$ Fisher-z Connectivity")
        
        degree_hc = np.sum(np.abs(mean_hc_fc), axis=1)
        degree_scz = np.sum(np.abs(mean_scz_fc), axis=1)
        diff_degree = degree_scz - degree_hc
        
        top_indices = np.argsort(np.abs(diff_degree))[-15:]
        top_diff = diff_degree[top_indices]
        parcels_labels = [f"ROI {idx+1}" for idx in top_indices]
        
        bar_colors = ["#D7191C" if d > 0 else "#2B83BA" for d in top_diff]
        axes[1].barh(parcels_labels, top_diff, color=bar_colors, edgecolor="black", linewidth=0.6)
        axes[1].axvline(0, color="black", linestyle="-", linewidth=1.0)
        axes[1].set_xlabel(r"Differential Nodal Degree Centrality ($\Delta$ Degree)", fontweight="bold")
        axes[1].set_title("Top 15 Most Altered Regional Hubs\n(Red: Hyperconnectivity, Blue: Hypoconnectivity)", fontweight="bold", pad=10)
        axes[1].grid(axis="x", linestyle=":", alpha=0.5)
        
        plt.tight_layout()
        fig6_path = fig_dir / "fig6_connectome_topological_biomarkers.png"
        plt.savefig(fig6_path, dpi=300)
        plt.savefig(fig_dir / "fig6_connectome_topological_biomarkers.pdf")
        plt.close()
        print(f"Saved: {fig6_path}")

    print("All 6 publication figures successfully generated in:", fig_dir)

if __name__ == "__main__":
    generate_all_figures()
