#!/usr/bin/env python3
"""
Advanced Visualizations for fMRI Quality Metrics
Generates radar charts, CDF plots, QQ plots, temporal plots, and more.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
warnings.filterwarnings('ignore')


class AdvancedVisualizations:
    """
    Advanced visualization methods for fMRI quality metrics.
    """
    
    def __init__(self, metrics_file: str, output_dir: str):
        """
        Initialize advanced visualizations.
        
        Parameters
        ----------
        metrics_file : str
            Path to quality metrics CSV file
        output_dir : str
            Output directory for plots
        """
        self.df = pd.read_csv(metrics_file)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        (self.output_dir / 'dpi300').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'dpi600').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'pdf').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'svg').mkdir(parents=True, exist_ok=True)
        
        self.setup_style()
    
    def setup_style(self):
        """Configure plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
        })
    
    def save_figure(self, fig, name: str):
        """Save figure in multiple formats."""
        fig.savefig(self.output_dir / 'dpi300' / f'{name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / 'dpi600' / f'{name}.png', dpi=600, bbox_inches='tight')
        fig.savefig(self.output_dir / 'pdf' / f'{name}.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(self.output_dir / 'svg' / f'{name}.svg', format='svg', bbox_inches='tight')
        plt.close(fig)
    
    def plot_radar_chart(self):
        """Generate radar charts comparing raw vs preprocessed metrics."""
        # Normalize metrics for radar chart
        metrics_for_radar = ['snr', 'tsnr', 'mean_signal', 'global_mean', 'entropy']
        
        raw_cols = [f'raw_{m}' for m in metrics_for_radar]
        preproc_cols = [f'preproc_{m}' for m in metrics_for_radar]
        
        # Check which columns exist
        available_raw = [c for c in raw_cols if c in self.df.columns]
        available_preproc = [c for c in preproc_cols if c in self.df.columns]
        
        if not available_raw or not available_preproc:
            print("Not enough metrics for radar chart")
            return
        
        # Compute mean values
        raw_means = [self.df[c].mean() for c in available_raw]
        preproc_means = [self.df[c].mean() for c in available_preproc]
        
        # Normalize to 0-1 range
        all_values = raw_means + preproc_means
        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        raw_normalized = [(v - min_val) / range_val for v in raw_means]
        preproc_normalized = [(v - min_val) / range_val for v in preproc_means]
        
        # Create radar chart
        labels = [c.replace('raw_', '').replace('_', ' ').title() for c in available_raw]
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        raw_normalized += raw_normalized[:1]
        preproc_normalized += preproc_normalized[:1]
        
        ax.plot(angles, raw_normalized, 'o-', linewidth=2, label='Raw', color='#E74C3C')
        ax.fill(angles, raw_normalized, alpha=0.25, color='#E74C3C')
        
        ax.plot(angles, preproc_normalized, 'o-', linewidth=2, label='Preprocessed', color='#3498DB')
        ax.fill(angles, preproc_normalized, alpha=0.25, color='#3498DB')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title('Quality Metrics Comparison: Raw vs Preprocessed', size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        self.save_figure(fig, 'radar_chart_comparison')
    
    def plot_cdf(self):
        """Generate Cumulative Distribution Function plots."""
        # Use metrics that have actual values (psnr/ssim are NaN for different spaces)
        metrics = ['snr', 'tsnr', 'mean_signal', 'global_std']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col in self.df.columns:
                raw_data = np.sort(self.df[raw_col].dropna())
                yvals = np.arange(1, len(raw_data) + 1) / len(raw_data)
                ax.plot(raw_data, yvals, linewidth=2, color='#E74C3C', label='Raw')
            
            if preproc_col in self.df.columns:
                preproc_data = np.sort(self.df[preproc_col].dropna())
                yvals = np.arange(1, len(preproc_data) + 1) / len(preproc_data)
                ax.plot(preproc_data, yvals, linewidth=2, color='#3498DB', label='Preprocessed')
            
            ax.set_xlabel(metric.replace('_', ' ').upper())
            ax.set_title(f'CDF: {metric.replace("_", " ").title()}')
            ax.legend()
            
            ax.set_ylabel('Cumulative Probability')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'cdf_plots')
    
    def plot_qq(self):
        """Generate QQ plots for normality assessment."""
        metrics = ['raw_snr', 'preproc_snr', 'raw_tsnr', 'preproc_tsnr']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                
                if len(data) > 1:
                    stats.probplot(data, dist="norm", plot=ax)
                    ax.set_title(f'QQ Plot: {metric.replace("_", " ").title()}')
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'qq_plots')
    
    def plot_violin_boxplot_combined(self):
        """Generate combined violin and boxplot figures."""
        metrics = ['snr', 'tsnr', 'mean_signal']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col in self.df.columns and preproc_col in self.df.columns:
                plot_df = pd.DataFrame({
                    'Condition': ['Raw'] * len(self.df[raw_col].dropna()) + 
                                ['Preprocessed'] * len(self.df[preproc_col].dropna()),
                    metric.replace('_', ' ').title(): pd.concat([
                        self.df[raw_col].dropna(), 
                        self.df[preproc_col].dropna()
                    ])
                })
                
                # Violin plot
                sns.violinplot(data=plot_df, x='Condition', y=metric.replace('_', ' ').title(),
                              palette=['#E74C3C', '#3498DB'], ax=ax, inner=None)
                
                # Box plot overlay
                sns.boxplot(data=plot_df, x='Condition', y=metric.replace('_', ' ').title(),
                           width=0.3, ax=ax, boxprops={'zorder': 2, 'facecolor': 'white'})
                
                ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'violin_boxplot_combined')
    
    def plot_pairwise_correlation_matrix(self):
        """Generate pairwise correlation matrix with significance."""
        # Select key metrics (excluding psnr/ssim which are NaN)
        key_metrics = ['raw_snr', 'preproc_snr', 'raw_tsnr', 'preproc_tsnr', 
                      'mean_fd', 'mean_dvars', 'raw_cnr', 'preproc_cnr']
        
        available_metrics = [m for m in key_metrics if m in self.df.columns]
        
        if len(available_metrics) < 2:
            return
        
        subset_df = self.df[available_metrics].dropna()
        
        # Compute correlation matrix
        corr_matrix = subset_df.corr()
        
        # Compute p-values
        from scipy.stats import pearsonr
        pval_matrix = pd.DataFrame(np.zeros((len(available_metrics), len(available_metrics))),
                                   columns=available_metrics, index=available_metrics)
        
        for i, m1 in enumerate(available_metrics):
            for j, m2 in enumerate(available_metrics):
                if i != j:
                    valid = subset_df[[m1, m2]].dropna()
                    if len(valid) > 2:
                        _, pval = pearsonr(valid[m1], valid[m2])
                        pval_matrix.loc[m1, m2] = pval
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create annotation with correlation and significance
        annot = corr_matrix.copy()
        for i in range(len(available_metrics)):
            for j in range(len(available_metrics)):
                if i != j:
                    pval = pval_matrix.iloc[i, j]
                    if pval < 0.001:
                        annot.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}***"
                    elif pval < 0.01:
                        annot.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}**"
                    elif pval < 0.05:
                        annot.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}*"
                    else:
                        annot.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}"
                else:
                    annot.iloc[i, j] = "1.00"
        
        sns.heatmap(corr_matrix, annot=annot, fmt='', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={'shrink': 0.8})
        
        ax.set_title('Pairwise Correlation Matrix\n(* p<0.05, ** p<0.01, *** p<0.001)')
        
        plt.tight_layout()
        self.save_figure(fig, 'pairwise_correlation_significance')
    
    def plot_bar_charts(self):
        """Generate bar charts for key metrics."""
        metrics = ['snr', 'tsnr']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col in self.df.columns and preproc_col in self.df.columns:
                means = [self.df[raw_col].mean(), self.df[preproc_col].mean()]
                stds = [self.df[raw_col].std(), self.df[preproc_col].std()]
                
                x = np.arange(2)
                bars = ax.bar(x, means, yerr=stds, capsize=5, 
                             color=['#E74C3C', '#3498DB'], alpha=0.7)
                
                ax.set_xticks(x)
                ax.set_xticklabels(['Raw', 'Preprocessed'])
                ax.set_ylabel(metric.upper())
                ax.set_title(f'Mean {metric.upper()} Comparison')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, mean, std in zip(bars, means, stds):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                           f'{mean:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        self.save_figure(fig, 'bar_chart_comparison')
    
    def plot_line_charts(self):
        """Generate line charts showing metric progression."""
        # Sort by subject number
        df_sorted = self.df.sort_values('subject')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = [('raw_snr', 'preproc_snr', 'SNR'),
                   ('raw_tsnr', 'preproc_tsnr', 'tSNR'),
                   ('raw_global_mean', 'preproc_global_mean', 'Global Mean'),
                   ('raw_entropy', 'preproc_entropy', 'Entropy')]
        
        for idx, (raw_col, preproc_col, title) in enumerate(metrics):
            ax = axes[idx]
            
            if raw_col in df_sorted.columns and preproc_col in df_sorted.columns:
                x = range(len(df_sorted))
                ax.plot(x, df_sorted[raw_col], 'o-', color='#E74C3C', 
                       label='Raw', linewidth=2, markersize=6)
                ax.plot(x, df_sorted[preproc_col], 's-', color='#3498DB', 
                       label='Preprocessed', linewidth=2, markersize=6)
                
                ax.set_xlabel('Subject')
                ax.set_ylabel(title)
                ax.set_title(f'{title} Across Subjects')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Set x-ticks
                ax.set_xticks(x[::5])
                ax.set_xticklabels(df_sorted['subject'].values[::5], rotation=45, ha='right')
        
        plt.tight_layout()
        self.save_figure(fig, 'line_chart_progression')
    
    def plot_motion_trace(self):
        """Generate motion trace plots."""
        motion_metrics = ['mean_fd', 'max_fd', 'std_fd', 'pct_fd_gt_02']
        
        available = [m for m in motion_metrics if m in self.df.columns]
        
        if not available:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        df_sorted = self.df.sort_values('subject')
        
        for idx, metric in enumerate(available[:4]):
            ax = axes[idx]
            
            if metric in df_sorted.columns:
                ax.bar(range(len(df_sorted)), df_sorted[metric].fillna(0), 
                      color='#3498DB', alpha=0.7)
                ax.axhline(df_sorted[metric].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {df_sorted[metric].mean():.3f}')
                
                ax.set_xlabel('Subject')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} by Subject')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'motion_trace')
    
    def plot_temporal_metrics(self):
        """Generate temporal metrics visualization."""
        temporal_metrics = ['temporal_variance', 'temporal_std', 'temporal_entropy', 
                           'num_outliers', 'pct_outliers', 'temporal_drift']
        
        raw_cols = [f'raw_{m}' for m in temporal_metrics]
        preproc_cols = [f'preproc_{m}' for m in temporal_metrics]
        
        available_raw = [c for c in raw_cols if c in self.df.columns]
        available_preproc = [c for c in preproc_cols if c in self.df.columns]
        
        if not available_raw and not available_preproc:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(temporal_metrics[:6]):
            ax = axes[idx]
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col in self.df.columns and preproc_col in self.df.columns:
                plot_df = pd.DataFrame({
                    'Condition': ['Raw'] * len(self.df[raw_col].dropna()) + 
                                ['Preprocessed'] * len(self.df[preproc_col].dropna()),
                    metric.replace('_', ' ').title(): pd.concat([
                        self.df[raw_col].dropna(), 
                        self.df[preproc_col].dropna()
                    ])
                })
                
                sns.boxplot(data=plot_df, x='Condition', y=metric.replace('_', ' ').title(),
                           palette=['#E74C3C', '#3498DB'], ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'temporal_metrics')
    
    def generate_all_plots(self):
        """Generate all advanced visualizations."""
        print("Generating advanced visualizations...")
        
        self.plot_radar_chart()
        self.plot_cdf()
        self.plot_qq()
        self.plot_violin_boxplot_combined()
        self.plot_pairwise_correlation_matrix()
        self.plot_bar_charts()
        self.plot_line_charts()
        self.plot_motion_trace()
        self.plot_temporal_metrics()
        
        print("Advanced visualizations complete!")


def main():
    """Main function."""
    metrics_file = '/root/fMRI/quality_analysis/metrics/quality_metrics.csv'
    output_dir = '/root/fMRI/quality_analysis/plots'
    
    viz = AdvancedVisualizations(metrics_file, output_dir)
    viz.generate_all_plots()


if __name__ == '__main__':
    main()
