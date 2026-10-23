#!/usr/bin/env python3
"""
Generate comprehensive signal quality plots for fMRI data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
data_path = Path('/root/fMRI/quality_analysis/signal_quality/signal_quality_metrics.csv')
output_dir = Path('/root/fMRI/quality_analysis/signal_quality_plots')
output_dir.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} subjects")

# Create subdirectories for different formats
(output_dir / 'png').mkdir(exist_ok=True)
(output_dir / 'pdf').mkdir(exist_ok=True)
(output_dir / 'svg').mkdir(exist_ok=True)

def save_figure(fig, name):
    """Save figure in multiple formats."""
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / fmt / f'{name}.{fmt}', dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# 1. SNR and TSNR Distributions
# =============================================================================
print("Generating SNR/TSNR plots...")

# SNR comparison: Raw vs Preprocessed
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw SNR
ax1 = axes[0]
sns.histplot(df['raw_snr'], kde=True, ax=ax1, color='steelblue', alpha=0.7)
ax1.axvline(df['raw_snr'].mean(), color='red', linestyle='--', label=f'Mean: {df["raw_snr"].mean():.2f}')
ax1.axvline(df['raw_snr'].median(), color='green', linestyle=':', label=f'Median: {df["raw_snr"].median():.2f}')
ax1.set_xlabel('Raw SNR', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Raw Signal-to-Noise Ratio Distribution', fontsize=14, fontweight='bold')
ax1.legend()

# Preprocessed SNR
ax2 = axes[1]
sns.histplot(df['preproc_snr'], kde=True, ax=ax2, color='darkorange', alpha=0.7)
ax2.axvline(df['preproc_snr'].mean(), color='red', linestyle='--', label=f'Mean: {df["preproc_snr"].mean():.2f}')
ax2.axvline(df['preproc_snr'].median(), color='green', linestyle=':', label=f'Median: {df["preproc_snr"].median():.2f}')
ax2.set_xlabel('Preprocessed SNR', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Preprocessed Signal-to-Noise Ratio Distribution', fontsize=14, fontweight='bold')
ax2.legend()

plt.tight_layout()
save_figure(fig, 'snr_distribution_comparison')

# TSNR comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Raw TSNR median
ax1 = axes[0, 0]
sns.histplot(df['raw_tsnr_median'], kde=True, ax=ax1, color='steelblue', alpha=0.7)
ax1.axvline(df['raw_tsnr_median'].mean(), color='red', linestyle='--', label=f'Mean: {df["raw_tsnr_median"].mean():.2f}')
ax1.set_xlabel('Raw TSNR (Median)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Raw Temporal SNR (Median)', fontsize=14, fontweight='bold')
ax1.legend()

# Raw TSNR mean
ax2 = axes[0, 1]
sns.histplot(df['raw_tsnr_mean'], kde=True, ax=ax2, color='steelblue', alpha=0.7)
ax2.axvline(df['raw_tsnr_mean'].mean(), color='red', linestyle='--', label=f'Mean: {df["raw_tsnr_mean"].mean():.2f}')
ax2.set_xlabel('Raw TSNR (Mean)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Raw Temporal SNR (Mean)', fontsize=14, fontweight='bold')
ax2.legend()

# Preprocessed TSNR median
ax3 = axes[1, 0]
sns.histplot(df['preproc_tsnr_median'], kde=True, ax=ax3, color='darkorange', alpha=0.7)
ax3.axvline(df['preproc_tsnr_median'].mean(), color='red', linestyle='--', label=f'Mean: {df["preproc_tsnr_median"].mean():.2f}')
ax3.set_xlabel('Preprocessed TSNR (Median)', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Preprocessed Temporal SNR (Median)', fontsize=14, fontweight='bold')
ax3.legend()

# Preprocessed TSNR mean
ax4 = axes[1, 1]
sns.histplot(df['preproc_tsnr_mean'], kde=True, ax=ax4, color='darkorange', alpha=0.7)
ax4.axvline(df['preproc_tsnr_mean'].mean(), color='red', linestyle='--', label=f'Mean: {df["preproc_tsnr_mean"].mean():.2f}')
ax4.set_xlabel('Preprocessed TSNR (Mean)', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Preprocessed Temporal SNR (Mean)', fontsize=14, fontweight='bold')
ax4.legend()

plt.tight_layout()
save_figure(fig, 'tsnr_distribution_comparison')

# =============================================================================
# 2. Boxplots and Violin Plots
# =============================================================================
print("Generating boxplots and violin plots...")

# SNR boxplot comparison
fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = pd.DataFrame({
    'Raw SNR': df['raw_snr'],
    'Preprocessed SNR': df['preproc_snr']
})
sns.boxplot(data=data_to_plot, ax=ax, palette=['steelblue', 'darkorange'])
ax.set_ylabel('SNR Value', fontsize=12)
ax.set_title('SNR Comparison: Raw vs Preprocessed', fontsize=14, fontweight='bold')
# Add statistical annotation
from scipy import stats
stat, pval = stats.ttest_rel(df['raw_snr'], df['preproc_snr'])
ax.text(0.5, 0.95, f'Paired t-test p={pval:.2e}', transform=ax.transAxes, 
        ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
save_figure(fig, 'snr_boxplot_comparison')

# TSNR violin plot
fig, ax = plt.subplots(figsize=(12, 6))
data_to_plot = pd.DataFrame({
    'Raw TSNR (Median)': df['raw_tsnr_median'],
    'Raw TSNR (Mean)': df['raw_tsnr_mean'],
    'Preproc TSNR (Median)': df['preproc_tsnr_median'],
    'Preproc TSNR (Mean)': df['preproc_tsnr_mean']
})
sns.violinplot(data=data_to_plot, ax=ax, palette='Set2')
ax.set_ylabel('TSNR Value', fontsize=12)
ax.set_title('Temporal SNR Distribution (Violin Plot)', fontsize=14, fontweight='bold')
plt.xticks(rotation=15)
save_figure(fig, 'tsnr_violin_comparison')

# =============================================================================
# 3. Subject-wise Analysis
# =============================================================================
print("Generating subject-wise plots...")

# SNR improvement per subject
fig, ax = plt.subplots(figsize=(16, 6))
df_sorted = df.sort_values('subject')
x = range(len(df_sorted))
width = 0.35
ax.bar([i - width/2 for i in x], df_sorted['raw_snr'], width, label='Raw SNR', color='steelblue', alpha=0.8)
ax.bar([i + width/2 for i in x], df_sorted['preproc_snr'], width, label='Preprocessed SNR', color='darkorange', alpha=0.8)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('SNR', fontsize=12)
ax.set_title('SNR by Subject: Raw vs Preprocessed', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_sorted['subject'], rotation=90, fontsize=7)
ax.legend()
ax.grid(axis='y', alpha=0.3)
save_figure(fig, 'snr_by_subject')

# TSNR improvement per subject
fig, ax = plt.subplots(figsize=(16, 6))
ax.bar([i - width/2 for i in x], df_sorted['raw_tsnr_median'], width, label='Raw TSNR', color='steelblue', alpha=0.8)
ax.bar([i + width/2 for i in x], df_sorted['preproc_tsnr_median'], width, label='Preprocessed TSNR', color='darkorange', alpha=0.8)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('TSNR (Median)', fontsize=12)
ax.set_title('Temporal SNR by Subject: Raw vs Preprocessed', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_sorted['subject'], rotation=90, fontsize=7)
ax.legend()
ax.grid(axis='y', alpha=0.3)
save_figure(fig, 'tsnr_by_subject')

# =============================================================================
# 4. Improvement Analysis
# =============================================================================
print("Generating improvement analysis plots...")

# Calculate improvements
df['snr_improvement'] = df['preproc_snr'] - df['raw_snr']
df['tsnr_improvement'] = df['preproc_tsnr_median'] - df['raw_tsnr_median']
df['snr_pct_improvement'] = (df['snr_improvement'] / df['raw_snr']) * 100
df['tsnr_pct_improvement'] = (df['tsnr_improvement'] / df['raw_tsnr_median']) * 100

# Improvement histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
sns.histplot(df['snr_improvement'], kde=True, ax=ax1, color='green', alpha=0.7)
ax1.axvline(0, color='red', linestyle='--', linewidth=2)
ax1.axvline(df['snr_improvement'].mean(), color='blue', linestyle=':', label=f'Mean: {df["snr_improvement"].mean():.3f}')
ax1.set_xlabel('SNR Improvement (Preprocessed - Raw)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('SNR Improvement After Preprocessing', fontsize=14, fontweight='bold')
ax1.legend()

ax2 = axes[1]
sns.histplot(df['tsnr_improvement'], kde=True, ax=ax2, color='green', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.axvline(df['tsnr_improvement'].mean(), color='blue', linestyle=':', label=f'Mean: {df["tsnr_improvement"].mean():.3f}')
ax2.set_xlabel('TSNR Improvement (Preprocessed - Raw)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Temporal SNR Improvement After Preprocessing', fontsize=14, fontweight='bold')
ax2.legend()

plt.tight_layout()
save_figure(fig, 'improvement_histograms')

# Scatter plot: Raw vs Preprocessed
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.scatter(df['raw_snr'], df['preproc_snr'], alpha=0.7, c='steelblue', edgecolors='black', linewidth=0.5)
min_val = min(df['raw_snr'].min(), df['preproc_snr'].min())
max_val = max(df['raw_snr'].max(), df['preproc_snr'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (No change)')
ax1.set_xlabel('Raw SNR', fontsize=12)
ax1.set_ylabel('Preprocessed SNR', fontsize=12)
ax1.set_title('SNR: Raw vs Preprocessed', fontsize=14, fontweight='bold')
ax1.legend()
# Add correlation
corr = df['raw_snr'].corr(df['preproc_snr'])
ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, fontsize=12, va='top')

ax2 = axes[1]
ax2.scatter(df['raw_tsnr_median'], df['preproc_tsnr_median'], alpha=0.7, c='darkorange', edgecolors='black', linewidth=0.5)
min_val = min(df['raw_tsnr_median'].min(), df['preproc_tsnr_median'].min())
max_val = max(df['raw_tsnr_median'].max(), df['preproc_tsnr_median'].max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (No change)')
ax2.set_xlabel('Raw TSNR (Median)', fontsize=12)
ax2.set_ylabel('Preprocessed TSNR (Median)', fontsize=12)
ax2.set_title('Temporal SNR: Raw vs Preprocessed', fontsize=14, fontweight='bold')
ax2.legend()
corr = df['raw_tsnr_median'].corr(df['preproc_tsnr_median'])
ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes, fontsize=12, va='top')

plt.tight_layout()
save_figure(fig, 'raw_vs_preproc_scatter')

# =============================================================================
# 5. Quality Metrics Correlation
# =============================================================================
print("Generating correlation plots...")

# Select key metrics for correlation
metrics = ['raw_snr', 'preproc_snr', 'raw_tsnr_median', 'preproc_tsnr_median', 
           'psnr', 'ssim', 'mse', 'rmse']
corr_matrix = df[metrics].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Quality Metrics Correlation Matrix', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
save_figure(fig, 'quality_metrics_correlation')

# =============================================================================
# 6. Image Quality Metrics
# =============================================================================
print("Generating image quality plots...")

# PSNR, SSIM, MSE, RMSE
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
sns.histplot(df['psnr'], kde=True, ax=ax1, color='purple', alpha=0.7)
ax1.axvline(df['psnr'].mean(), color='red', linestyle='--', label=f'Mean: {df["psnr"].mean():.2f}')
ax1.set_xlabel('PSNR (dB)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Peak Signal-to-Noise Ratio (PSNR)', fontsize=14, fontweight='bold')
ax1.legend()

ax2 = axes[0, 1]
sns.histplot(df['ssim'], kde=True, ax=ax2, color='teal', alpha=0.7)
ax2.axvline(df['ssim'].mean(), color='red', linestyle='--', label=f'Mean: {df["ssim"].mean():.4f}')
ax2.set_xlabel('SSIM', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Structural Similarity Index (SSIM)', fontsize=14, fontweight='bold')
ax2.legend()

ax3 = axes[1, 0]
sns.histplot(df['mse'], kde=True, ax=ax3, color='coral', alpha=0.7)
ax3.axvline(df['mse'].mean(), color='red', linestyle='--', label=f'Mean: {df["mse"].mean():.2f}')
ax3.set_xlabel('MSE', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
ax3.legend()

ax4 = axes[1, 1]
sns.histplot(df['rmse'], kde=True, ax=ax4, color='olive', alpha=0.7)
ax4.axvline(df['rmse'].mean(), color='red', linestyle='--', label=f'Mean: {df["rmse"].mean():.2f}')
ax4.set_xlabel('RMSE', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Root Mean Squared Error (RMSE)', fontsize=14, fontweight='bold')
ax4.legend()

plt.tight_layout()
save_figure(fig, 'image_quality_metrics')

# =============================================================================
# 7. Signal Statistics
# =============================================================================
print("Generating signal statistics plots...")

# Mean signal inside vs outside brain
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
sns.histplot(df['mean_inside'], kde=True, ax=ax1, color='green', alpha=0.7, label='Inside Brain')
sns.histplot(df['mean_outside'], kde=True, ax=ax1, color='red', alpha=0.5, label='Outside Brain')
ax1.set_xlabel('Mean Signal Intensity', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Mean Signal: Inside vs Outside Brain', fontsize=14, fontweight='bold')
ax1.legend()

ax2 = axes[1]
df['signal_ratio'] = df['mean_inside'] / df['mean_outside']
sns.histplot(df['signal_ratio'], kde=True, ax=ax2, color='purple', alpha=0.7)
ax2.axvline(df['signal_ratio'].mean(), color='red', linestyle='--', label=f'Mean: {df["signal_ratio"].mean():.2f}')
ax2.set_xlabel('Signal Ratio (Inside/Outside)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Signal Ratio (Inside/Outside Brain)', fontsize=14, fontweight='bold')
ax2.legend()

plt.tight_layout()
save_figure(fig, 'signal_statistics')

# =============================================================================
# 8. Brain Coverage Analysis
# =============================================================================
print("Generating brain coverage plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
sns.histplot(df['brain_coverage_mean'], kde=True, ax=ax1, color='navy', alpha=0.7)
ax1.axvline(df['brain_coverage_mean'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["brain_coverage_mean"].mean():.4f}')
ax1.set_xlabel('Brain Coverage (Mean)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Brain Coverage Distribution', fontsize=14, fontweight='bold')
ax1.legend()

ax2 = axes[1]
sns.histplot(df['non_zero_voxels'], kde=True, ax=ax2, color='darkgreen', alpha=0.7)
ax2.axvline(df['non_zero_voxels'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["non_zero_voxels"].mean():.0f}')
ax2.set_xlabel('Non-Zero Voxels', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Non-Zero Voxels Distribution', fontsize=14, fontweight='bold')
ax2.legend()

plt.tight_layout()
save_figure(fig, 'brain_coverage')

# =============================================================================
# 9. Summary Dashboard
# =============================================================================
print("Generating summary dashboard...")

fig = plt.figure(figsize=(20, 16))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: SNR comparison
ax1 = fig.add_subplot(gs[0, 0])
data = pd.DataFrame({'Raw': df['raw_snr'], 'Preprocessed': df['preproc_snr']})
sns.boxplot(data=data, ax=ax1, palette=['steelblue', 'darkorange'])
ax1.set_title('SNR Comparison', fontsize=12, fontweight='bold')
ax1.set_ylabel('SNR')

# Panel 2: TSNR comparison
ax2 = fig.add_subplot(gs[0, 1])
data = pd.DataFrame({'Raw': df['raw_tsnr_median'], 'Preprocessed': df['preproc_tsnr_median']})
sns.boxplot(data=data, ax=ax2, palette=['steelblue', 'darkorange'])
ax2.set_title('TSNR Comparison', fontsize=12, fontweight='bold')
ax2.set_ylabel('TSNR')

# Panel 3: SSIM distribution
ax3 = fig.add_subplot(gs[0, 2])
sns.histplot(df['ssim'], kde=True, ax=ax3, color='teal', alpha=0.7)
ax3.axvline(df['ssim'].mean(), color='red', linestyle='--')
ax3.set_title('SSIM Distribution', fontsize=12, fontweight='bold')
ax3.set_xlabel('SSIM')

# Panel 4: SNR improvement
ax4 = fig.add_subplot(gs[1, 0])
sns.histplot(df['snr_improvement'], kde=True, ax=ax4, color='green', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.set_title('SNR Improvement', fontsize=12, fontweight='bold')
ax4.set_xlabel('Improvement')

# Panel 5: TSNR improvement
ax5 = fig.add_subplot(gs[1, 1])
sns.histplot(df['tsnr_improvement'], kde=True, ax=ax5, color='green', alpha=0.7)
ax5.axvline(0, color='red', linestyle='--', linewidth=2)
ax5.set_title('TSNR Improvement', fontsize=12, fontweight='bold')
ax5.set_xlabel('Improvement')

# Panel 6: PSNR distribution
ax6 = fig.add_subplot(gs[1, 2])
sns.histplot(df['psnr'], kde=True, ax=ax6, color='purple', alpha=0.7)
ax6.axvline(df['psnr'].mean(), color='red', linestyle='--')
ax6.set_title('PSNR Distribution', fontsize=12, fontweight='bold')
ax6.set_xlabel('PSNR (dB)')

# Panel 7: Raw vs Preproc SNR scatter
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(df['raw_snr'], df['preproc_snr'], alpha=0.6, c='steelblue')
ax7.plot([df['raw_snr'].min(), df['raw_snr'].max()], 
         [df['raw_snr'].min(), df['raw_snr'].max()], 'r--')
ax7.set_xlabel('Raw SNR')
ax7.set_ylabel('Preprocessed SNR')
ax7.set_title('SNR: Raw vs Preprocessed', fontsize=12, fontweight='bold')

# Panel 8: Raw vs Preproc TSNR scatter
ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(df['raw_tsnr_median'], df['preproc_tsnr_median'], alpha=0.6, c='darkorange')
ax8.plot([df['raw_tsnr_median'].min(), df['raw_tsnr_median'].max()], 
         [df['raw_tsnr_median'].min(), df['raw_tsnr_median'].max()], 'r--')
ax8.set_xlabel('Raw TSNR')
ax8.set_ylabel('Preprocessed TSNR')
ax8.set_title('TSNR: Raw vs Preprocessed', fontsize=12, fontweight='bold')

# Panel 9: Summary statistics
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
summary_text = f"""
Summary Statistics
{'='*40}
Number of Subjects: {len(df)}

SNR Statistics:
  Raw SNR:        Mean={df['raw_snr'].mean():.2f}, SD={df['raw_snr'].std():.2f}
  Preproc SNR:    Mean={df['preproc_snr'].mean():.2f}, SD={df['preproc_snr'].std():.2f}
  Improvement:    Mean={df['snr_improvement'].mean():.3f}

TSNR Statistics:
  Raw TSNR:       Mean={df['raw_tsnr_median'].mean():.2f}, SD={df['raw_tsnr_median'].std():.2f}
  Preproc TSNR:   Mean={df['preproc_tsnr_median'].mean():.2f}, SD={df['preproc_tsnr_median'].std():.2f}
  Improvement:    Mean={df['tsnr_improvement'].mean():.3f}

Image Quality:
  PSNR:           Mean={df['psnr'].mean():.2f} dB
  SSIM:           Mean={df['ssim'].mean():.4f}
  RMSE:           Mean={df['rmse'].mean():.2f}

Brain Coverage:   Mean={df['brain_coverage_mean'].mean():.4f}
"""
ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('fMRI Signal Quality Dashboard', fontsize=18, fontweight='bold', y=0.98)
save_figure(fig, 'quality_dashboard')

# =============================================================================
# 10. Save Summary Statistics
# =============================================================================
print("Saving summary statistics...")

summary_stats = {
    'n_subjects': len(df),
    'snr': {
        'raw': {'mean': df['raw_snr'].mean(), 'std': df['raw_snr'].std(), 'min': df['raw_snr'].min(), 'max': df['raw_snr'].max()},
        'preproc': {'mean': df['preproc_snr'].mean(), 'std': df['preproc_snr'].std(), 'min': df['preproc_snr'].min(), 'max': df['preproc_snr'].max()},
        'improvement': {'mean': df['snr_improvement'].mean(), 'std': df['snr_improvement'].std()}
    },
    'tsnr': {
        'raw': {'mean': df['raw_tsnr_median'].mean(), 'std': df['raw_tsnr_median'].std()},
        'preproc': {'mean': df['preproc_tsnr_median'].mean(), 'std': df['preproc_tsnr_median'].std()},
        'improvement': {'mean': df['tsnr_improvement'].mean(), 'std': df['tsnr_improvement'].std()}
    },
    'image_quality': {
        'psnr': {'mean': df['psnr'].mean(), 'std': df['psnr'].std()},
        'ssim': {'mean': df['ssim'].mean(), 'std': df['ssim'].std()},
        'rmse': {'mean': df['rmse'].mean(), 'std': df['rmse'].std()}
    }
}

import json
with open(output_dir / 'summary_statistics.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"\nAll plots saved to: {output_dir}")
print("Formats: png, pdf, svg")
print("\nGenerated plots:")
for f in sorted((output_dir / 'png').glob('*.png')):
    print(f"  - {f.name}")
