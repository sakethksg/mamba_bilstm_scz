#!/usr/bin/env python3
"""
Comprehensive fMRI Quality Metrics Framework
Computes quality metrics before and after preprocessing, generates visualizations,
and performs statistical analysis.
"""

import os
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from skimage.metrics import structural_similarity as ssim
import json
from datetime import datetime
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')


class FMRIQualityMetrics:
    """
    Comprehensive fMRI quality metrics computation framework.
    """
    
    def __init__(self, 
                 raw_dir: str,
                 preproc_dir: str,
                 output_dir: str,
                 n_jobs: int = -1):
        """
        Initialize the quality metrics framework.
        
        Parameters
        ----------
        raw_dir : str
            Path to raw BIDS dataset
        preproc_dir : str
            Path to preprocessed dataset (fMRIPrep output)
        output_dir : str
            Path to output directory for results
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        """
        self.raw_dir = Path(raw_dir)
        self.preproc_dir = Path(preproc_dir)
        self.output_dir = Path(output_dir)
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots' / 'dpi300').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots' / 'dpi600').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots' / 'pdf').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots' / 'svg').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'statistics').mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """Configure publication-quality plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.linewidth': 1.5,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2,
            'lines.markersize': 8,
        })
        
    def find_subjects(self) -> List[str]:
        """Find all subjects in the dataset."""
        subjects = []
        for item in self.raw_dir.iterdir():
            if item.is_dir() and item.name.startswith('sub-'):
                subjects.append(item.name)
        return sorted(subjects)
    
    def load_bold_data(self, subject: str, space: str = 'MNI152NLin2009cAsym') -> Tuple[Optional[nib.Nifti1Image], Optional[nib.Nifti1Image]]:
        """
        Load raw and preprocessed BOLD data for a subject.
        
        Parameters
        ----------
        subject : str
            Subject ID (e.g., 'sub-01')
        space : str
            Space of preprocessed data
            
        Returns
        -------
        raw_img, preproc_img : tuple of nibabel images
        """
        # Find raw BOLD file
        raw_func_dir = self.raw_dir / subject / 'func'
        raw_bold_files = list(raw_func_dir.glob('*_bold.nii.gz'))
        
        if not raw_bold_files:
            print(f"No raw BOLD file found for {subject}")
            return None, None
        
        raw_bold_file = raw_bold_files[0]
        
        # Find preprocessed BOLD file
        preproc_func_dir = self.preproc_dir / subject / 'func'
        # Try different naming patterns (with res-2 or without)
        preproc_bold_files = list(preproc_func_dir.glob(f'*_space-{space}_*_desc-preproc_bold.nii.gz'))
        if not preproc_bold_files:
            preproc_bold_files = list(preproc_func_dir.glob(f'*_space-{space}_desc-preproc_bold.nii.gz'))
        
        if not preproc_bold_files:
            print(f"No preprocessed BOLD file found for {subject}")
            return None, None
        
        preproc_bold_file = preproc_bold_files[0]
        
        # Load images
        try:
            raw_img = nib.load(str(raw_bold_file))
            preproc_img = nib.load(str(preproc_bold_file))
            return raw_img, preproc_img
        except Exception as e:
            print(f"Error loading images for {subject}: {e}")
            return None, None
    
    def load_confounds(self, subject: str) -> Optional[pd.DataFrame]:
        """Load confounds file for a subject."""
        confounds_file = self.preproc_dir / subject / 'func' / f'{subject}_task-speech_desc-confounds_timeseries.tsv'
        
        if not confounds_file.exists():
            return None
        
        return pd.read_csv(confounds_file, sep='\t')
    
    def compute_snr(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Signal-to-Noise Ratio.
        
        SNR = mean(signal) / std(noise)
        """
        if mask is not None:
            signal_data = data[mask > 0]
        else:
            signal_data = data[data > 0]
        
        if len(signal_data) == 0:
            return np.nan
        
        mean_signal = np.mean(signal_data)
        noise_std = np.std(signal_data)
        
        if noise_std == 0:
            return np.inf
        
        return mean_signal / noise_std
    
    def compute_tsnr(self, data: np.ndarray) -> float:
        """
        Compute Temporal Signal-to-Noise Ratio.
        
        tSNR = mean(time series) / std(time series)
        """
        if data.ndim < 4:
            return np.nan
        
        # Compute mean and std across time
        mean_ts = np.mean(data, axis=-1)
        std_ts = np.std(data, axis=-1)
        
        # Avoid division by zero
        std_ts[std_ts == 0] = np.nan
        
        tsnr_map = mean_ts / std_ts
        tsnr_map = tsnr_map[~np.isnan(tsnr_map)]
        
        if len(tsnr_map) == 0:
            return np.nan
        
        return np.mean(tsnr_map)
    
    def compute_basic_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute basic signal metrics."""
        # Flatten spatial dimensions
        if data.ndim == 4:
            # For 4D data, compute metrics on mean volume
            mean_vol = np.mean(data, axis=-1)
            flat_data = mean_vol[mean_vol > 0]
        else:
            flat_data = data[data > 0]
        
        if len(flat_data) == 0:
            return {
                'mean_signal': np.nan,
                'signal_variance': np.nan,
                'noise_variance': np.nan,
                'background_noise': np.nan,
                'global_mean': np.nan,
                'global_std': np.nan,
            }
        
        return {
            'mean_signal': float(np.mean(flat_data)),
            'signal_variance': float(np.var(flat_data)),
            'noise_variance': float(np.var(flat_data[flat_data < np.percentile(flat_data, 25)])) if len(flat_data[flat_data < np.percentile(flat_data, 25)]) > 0 else np.nan,
            'background_noise': float(np.mean(flat_data[flat_data < np.percentile(flat_data, 10)])) if len(flat_data[flat_data < np.percentile(flat_data, 10)]) > 0 else np.nan,
            'global_mean': float(np.mean(flat_data)),
            'global_std': float(np.std(flat_data)),
        }
    
    def compute_image_similarity(self, raw_data: np.ndarray, preproc_data: np.ndarray) -> Dict[str, float]:
        """
        Compute image similarity metrics between raw and preprocessed data.
        Note: Raw and preprocessed data may be in different spaces, so we compute
        distribution-based metrics instead of voxel-wise metrics.
        """
        # Extract mean volume for 4D data
        if raw_data.ndim == 4:
            raw_mean = np.mean(raw_data, axis=-1)
        else:
            raw_mean = raw_data
            
        if preproc_data.ndim == 4:
            preproc_mean = np.mean(preproc_data, axis=-1)
        else:
            preproc_mean = preproc_data
        
        # Get non-zero voxels
        raw_flat = raw_mean.flatten()
        preproc_flat = preproc_mean.flatten()
        
        raw_valid = raw_flat[raw_flat != 0]
        preproc_valid = preproc_flat[preproc_flat != 0]
        
        if len(raw_valid) == 0 or len(preproc_valid) == 0:
            return {metric: np.nan for metric in ['psnr', 'ssim', 'nmse', 'mse', 'rmse', 'mae', 
                                                    'pearson_r', 'ncc', 'mi', 'nmi']}
        
        metrics = {}
        
        # Since images are in different spaces, compute distribution-based metrics
        # Compare statistical distributions instead of voxel-wise comparisons
        
        # Distribution statistics
        raw_mean_val = np.mean(raw_valid)
        raw_std_val = np.std(raw_valid)
        preproc_mean_val = np.mean(preproc_valid)
        preproc_std_val = np.std(preproc_valid)
        
        # Normalized difference in means (effect size)
        pooled_std = np.sqrt((raw_std_val**2 + preproc_std_val**2) / 2)
        if pooled_std > 0:
            metrics['mean_diff_cohen_d'] = float(abs(raw_mean_val - preproc_mean_val) / pooled_std)
        else:
            metrics['mean_diff_cohen_d'] = np.nan
        
        # Coefficient of variation comparison
        raw_cv = raw_std_val / raw_mean_val if raw_mean_val != 0 else np.nan
        preproc_cv = preproc_std_val / preproc_mean_val if preproc_mean_val != 0 else np.nan
        
        metrics['raw_cv'] = float(raw_cv) if not np.isnan(raw_cv) else np.nan
        metrics['preproc_cv'] = float(preproc_cv) if not np.isnan(preproc_cv) else np.nan
        
        # Histogram-based metrics
        try:
            # Compute histograms
            raw_hist, raw_bins = np.histogram(raw_valid, bins=100, density=True)
            preproc_hist, preproc_bins = np.histogram(preproc_valid, bins=100, density=True)
            
            # Histogram intersection (similarity)
            min_bins = min(len(raw_hist), len(preproc_hist))
            hist_intersection = np.sum(np.minimum(raw_hist[:min_bins], preproc_hist[:min_bins]))
            metrics['hist_intersection'] = float(hist_intersection)
            
            # Bhattacharyya coefficient
            bc = np.sum(np.sqrt(raw_hist[:min_bins] * preproc_hist[:min_bins]))
            metrics['bhattacharyya_coeff'] = float(bc)
            
        except Exception:
            metrics['hist_intersection'] = np.nan
            metrics['bhattacharyya_coeff'] = np.nan
        
        # Kolmogorov-Smirnov statistic
        try:
            ks_stat, ks_pval = stats.ks_2samp(raw_valid, preproc_valid)
            metrics['ks_statistic'] = float(ks_stat)
            metrics['ks_pvalue'] = float(ks_pval)
        except Exception:
            metrics['ks_statistic'] = np.nan
            metrics['ks_pvalue'] = np.nan
        
        # Wasserstein distance (Earth Mover's Distance)
        try:
            wasserstein = stats.wasserstein_distance(raw_valid, preproc_valid)
            metrics['wasserstein_distance'] = float(wasserstein)
        except Exception:
            metrics['wasserstein_distance'] = np.nan
        
        # Set traditional metrics to NaN since we can't compute them across spaces
        metrics['psnr'] = np.nan
        metrics['ssim'] = np.nan
        metrics['nmse'] = np.nan
        metrics['mse'] = np.nan
        metrics['rmse'] = np.nan
        metrics['mae'] = np.nan
        metrics['pearson_r'] = np.nan
        metrics['ncc'] = np.nan
        metrics['mi'] = np.nan
        metrics['nmi'] = np.nan
        
        return metrics
    
    def compute_motion_metrics(self, confounds: pd.DataFrame) -> Dict[str, float]:
        """Compute motion-related metrics from confounds."""
        metrics = {}
        
        # Framewise Displacement
        if 'framewise_displacement' in confounds.columns:
            fd = confounds['framewise_displacement'].dropna().values
            
            metrics['mean_fd'] = float(np.mean(fd))
            metrics['max_fd'] = float(np.max(fd))
            metrics['median_fd'] = float(np.median(fd))
            metrics['std_fd'] = float(np.std(fd))
            metrics['pct_fd_gt_02'] = float(np.sum(fd > 0.2) / len(fd) * 100)
            metrics['pct_fd_gt_05'] = float(np.sum(fd > 0.5) / len(fd) * 100)
            metrics['num_high_motion'] = int(np.sum(fd > 0.5))
        else:
            metrics.update({
                'mean_fd': np.nan, 'max_fd': np.nan, 'median_fd': np.nan,
                'std_fd': np.nan, 'pct_fd_gt_02': np.nan, 'pct_fd_gt_05': np.nan,
                'num_high_motion': np.nan
            })
        
        # DVARS
        if 'dvars' in confounds.columns:
            dvars = confounds['dvars'].dropna().values
            metrics['mean_dvars'] = float(np.mean(dvars))
            metrics['std_dvars'] = float(np.std(dvars))
        else:
            metrics['mean_dvars'] = np.nan
            metrics['std_dvars'] = np.nan
        
        # Translation and rotation
        trans_cols = ['trans_x', 'trans_y', 'trans_z']
        rot_cols = ['rot_x', 'rot_y', 'rot_z']
        
        if all(col in confounds.columns for col in trans_cols):
            trans_data = confounds[trans_cols].values
            metrics['total_translation'] = float(np.sum(np.abs(trans_data)))
        else:
            metrics['total_translation'] = np.nan
        
        if all(col in confounds.columns for col in rot_cols):
            rot_data = confounds[rot_cols].values
            metrics['total_rotation'] = float(np.sum(np.abs(rot_data)))
        else:
            metrics['total_rotation'] = np.nan
        
        return metrics
    
    def compute_temporal_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute temporal quality metrics."""
        metrics = {}
        
        if data.ndim != 4:
            return {k: np.nan for k in ['temporal_variance', 'temporal_std', 'temporal_entropy',
                                        'num_outliers', 'pct_outliers', 'temporal_drift']}
        
        # Compute temporal statistics
        temporal_mean = np.mean(data, axis=-1)
        temporal_std = np.std(data, axis=-1)
        temporal_var = np.var(data, axis=-1)
        
        metrics['temporal_variance'] = float(np.mean(temporal_var))
        metrics['temporal_std'] = float(np.mean(temporal_std))
        
        # Temporal entropy - entropy of temporal variability (not spatial)
        try:
            # Compute entropy of the temporal variance distribution
            # This measures the complexity of temporal fluctuations
            temporal_var_flat = temporal_var.flatten()
            temporal_var_valid = temporal_var_flat[temporal_var_flat > 0]
            hist, _ = np.histogram(temporal_var_valid, bins=50, density=True)
            hist = hist[hist > 0]
            metrics['temporal_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
        except:
            metrics['temporal_entropy'] = np.nan
        
        # Outlier detection (using IQR method)
        try:
            global_signal = np.mean(data, axis=(0, 1, 2))
            Q1 = np.percentile(global_signal, 25)
            Q3 = np.percentile(global_signal, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (global_signal < lower_bound) | (global_signal > upper_bound)
            metrics['num_outliers'] = int(np.sum(outliers))
            metrics['pct_outliers'] = float(np.sum(outliers) / len(global_signal) * 100)
        except:
            metrics['num_outliers'] = np.nan
            metrics['pct_outliers'] = np.nan
        
        # Temporal drift (linear trend)
        try:
            global_signal = np.mean(data, axis=(0, 1, 2))
            time_points = np.arange(len(global_signal))
            slope, _ = np.polyfit(time_points, global_signal, 1)
            metrics['temporal_drift'] = float(slope)
        except:
            metrics['temporal_drift'] = np.nan
        
        return metrics
    
    def compute_optional_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute optional quality metrics."""
        metrics = {}
        
        if data.ndim == 4:
            mean_vol = np.mean(data, axis=-1)
        else:
            mean_vol = data
        
        valid_data = mean_vol[mean_vol > 0]
        
        if len(valid_data) == 0:
            return {k: np.nan for k in ['cnr', 'entropy', 'efc', 'fber', 'cjv',
                                        'sharpness', 'gradient_magnitude', 'dynamic_range']}
        
        # Contrast-to-Noise Ratio
        try:
            foreground = valid_data[valid_data > np.percentile(valid_data, 75)]
            background = valid_data[valid_data < np.percentile(valid_data, 25)]
            if len(foreground) > 0 and len(background) > 0:
                metrics['cnr'] = float((np.mean(foreground) - np.mean(background)) / 
                                       (np.std(foreground) + np.std(background) + 1e-10))
            else:
                metrics['cnr'] = np.nan
        except:
            metrics['cnr'] = np.nan
        
        # Entropy
        try:
            hist, _ = np.histogram(valid_data, bins=50, density=True)
            hist = hist[hist > 0]
            metrics['entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
        except:
            metrics['entropy'] = np.nan
        
        # Entropy Focus Criterion
        try:
            metrics['efc'] = float(np.sum(valid_data ** 2) / (np.sum(valid_data) ** 2 + 1e-10))
        except:
            metrics['efc'] = np.nan
        
        # Foreground-Background Energy Ratio
        try:
            foreground_energy = np.sum(valid_data[valid_data > np.percentile(valid_data, 50)] ** 2)
            background_energy = np.sum(valid_data[valid_data <= np.percentile(valid_data, 50)] ** 2)
            metrics['fber'] = float(foreground_energy / (background_energy + 1e-10))
        except:
            metrics['fber'] = np.nan
        
        # Coefficient of Joint Variation
        try:
            metrics['cjv'] = float(np.std(valid_data) / (np.mean(valid_data) + 1e-10))
        except:
            metrics['cjv'] = np.nan
        
        # Image Sharpness (gradient-based)
        try:
            if mean_vol.ndim == 3:
                grad_x = np.gradient(mean_vol, axis=0)
                grad_y = np.gradient(mean_vol, axis=1)
                grad_z = np.gradient(mean_vol, axis=2)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
                metrics['sharpness'] = float(np.mean(grad_mag))
                metrics['gradient_magnitude'] = float(np.mean(grad_mag))
            else:
                metrics['sharpness'] = np.nan
                metrics['gradient_magnitude'] = np.nan
        except:
            metrics['sharpness'] = np.nan
            metrics['gradient_magnitude'] = np.nan
        
        # Dynamic Range
        try:
            metrics['dynamic_range'] = float(np.max(valid_data) - np.min(valid_data))
        except:
            metrics['dynamic_range'] = np.nan
        
        return metrics
    
    def process_subject(self, subject: str) -> Dict[str, float]:
        """Process a single subject and compute all metrics."""
        print(f"Processing {subject}...")
        
        results = {'subject': subject}
        
        # Load data
        raw_img, preproc_img = self.load_bold_data(subject)
        
        if raw_img is None or preproc_img is None:
            return results
        
        raw_data = raw_img.get_fdata()
        preproc_data = preproc_img.get_fdata()
        
        # Compute metrics for raw data
        results['raw_snr'] = self.compute_snr(raw_data)
        results['raw_tsnr'] = self.compute_tsnr(raw_data)
        raw_basic = self.compute_basic_metrics(raw_data)
        for k, v in raw_basic.items():
            results[f'raw_{k}'] = v
        
        raw_temporal = self.compute_temporal_metrics(raw_data)
        for k, v in raw_temporal.items():
            results[f'raw_{k}'] = v
        
        raw_optional = self.compute_optional_metrics(raw_data)
        for k, v in raw_optional.items():
            results[f'raw_{k}'] = v
        
        # Compute metrics for preprocessed data
        results['preproc_snr'] = self.compute_snr(preproc_data)
        results['preproc_tsnr'] = self.compute_tsnr(preproc_data)
        preproc_basic = self.compute_basic_metrics(preproc_data)
        for k, v in preproc_basic.items():
            results[f'preproc_{k}'] = v
        
        preproc_temporal = self.compute_temporal_metrics(preproc_data)
        for k, v in preproc_temporal.items():
            results[f'preproc_{k}'] = v
        
        preproc_optional = self.compute_optional_metrics(preproc_data)
        for k, v in preproc_optional.items():
            results[f'preproc_{k}'] = v
        
        # Compute image similarity metrics
        similarity = self.compute_image_similarity(raw_data, preproc_data)
        results.update(similarity)
        
        # Load confounds and compute motion metrics
        confounds = self.load_confounds(subject)
        if confounds is not None:
            motion_metrics = self.compute_motion_metrics(confounds)
            results.update(motion_metrics)
        
        # Compute before vs after differences
        for metric in ['snr', 'tsnr', 'mean_signal', 'signal_variance', 'global_mean', 'global_std']:
            raw_key = f'raw_{metric}'
            preproc_key = f'preproc_{metric}'
            if raw_key in results and preproc_key in results:
                raw_val = results[raw_key]
                preproc_val = results[preproc_key]
                
                if not np.isnan(raw_val) and not np.isnan(preproc_val):
                    results[f'{metric}_abs_diff'] = abs(preproc_val - raw_val)
                    results[f'{metric}_rel_diff'] = abs(preproc_val - raw_val) / (abs(raw_val) + 1e-10)
                    
                    if raw_val != 0:
                        results[f'{metric}_pct_change'] = ((preproc_val - raw_val) / abs(raw_val)) * 100
                        results[f'{metric}_pct_improvement'] = abs((preproc_val - raw_val) / abs(raw_val)) * 100
                    else:
                        results[f'{metric}_pct_change'] = np.nan
                        results[f'{metric}_pct_improvement'] = np.nan
        
        return results
    
    def run_analysis(self, subjects: Optional[List[str]] = None) -> pd.DataFrame:
        """Run quality metrics analysis on all subjects."""
        if subjects is None:
            subjects = self.find_subjects()
        
        print(f"Found {len(subjects)} subjects")
        
        # Process subjects in parallel
        with mp.Pool(self.n_jobs) as pool:
            results = pool.map(self.process_subject, subjects)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / 'metrics' / 'quality_metrics.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved metrics to {output_file}")
        
        return df
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute comprehensive statistics."""
        stats_dict = {}
        
        # Descriptive statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        desc_stats = df[numeric_cols].describe()
        
        # Add additional statistics
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                desc_stats.loc['variance', col] = np.var(data)
                desc_stats.loc['range', col] = np.max(data) - np.min(data)
                desc_stats.loc['iqr', col] = np.percentile(data, 75) - np.percentile(data, 25)
                desc_stats.loc['skewness', col] = stats.skew(data)
                desc_stats.loc['kurtosis', col] = stats.kurtosis(data)
                
                # 95% CI
                ci = stats.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=stats.sem(data))
                desc_stats.loc['ci_lower', col] = ci[0]
                desc_stats.loc['ci_upper', col] = ci[1]
        
        stats_dict['descriptive'] = desc_stats
        
        # Before vs After comparison statistics
        comparison_metrics = ['snr', 'tsnr', 'mean_signal', 'signal_variance', 'global_mean', 'global_std']
        comparison_results = []
        
        for metric in comparison_metrics:
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col in df.columns and preproc_col in df.columns:
                raw_vals = df[raw_col].dropna()
                preproc_vals = df[preproc_col].dropna()
                
                # Match subjects
                valid_idx = raw_vals.index.intersection(preproc_vals.index)
                raw_matched = raw_vals.loc[valid_idx]
                preproc_matched = preproc_vals.loc[valid_idx]
                
                if len(raw_matched) > 1:
                    # Paired t-test
                    t_stat, t_pval = stats.ttest_rel(raw_matched, preproc_matched)
                    
                    # Wilcoxon signed-rank test
                    try:
                        w_stat, w_pval = stats.wilcoxon(raw_matched, preproc_matched)
                    except:
                        w_stat, w_pval = np.nan, np.nan
                    
                    # Cohen's d
                    pooled_std = np.sqrt((np.std(raw_matched)**2 + np.std(preproc_matched)**2) / 2)
                    cohens_d = (np.mean(preproc_matched) - np.mean(raw_matched)) / (pooled_std + 1e-10)
                    
                    # Effect size (r)
                    effect_r = cohens_d / np.sqrt(cohens_d**2 + 4)
                    
                    comparison_results.append({
                        'metric': metric,
                        'raw_mean': np.mean(raw_matched),
                        'preproc_mean': np.mean(preproc_matched),
                        'raw_std': np.std(raw_matched),
                        'preproc_std': np.std(preproc_matched),
                        'abs_diff_mean': np.mean(np.abs(preproc_matched - raw_matched)),
                        'rel_diff_mean': np.mean(np.abs(preproc_matched - raw_matched) / (np.abs(raw_matched) + 1e-10)),
                        'pct_change_mean': np.mean(((preproc_matched - raw_matched) / (np.abs(raw_matched) + 1e-10)) * 100),
                        'paired_t_stat': t_stat,
                        'paired_t_pval': t_pval,
                        'wilcoxon_stat': w_stat,
                        'wilcoxon_pval': w_pval,
                        'cohens_d': cohens_d,
                        'effect_size_r': effect_r,
                        'n_subjects': len(raw_matched)
                    })
        
        if comparison_results:
            stats_dict['comparison'] = pd.DataFrame(comparison_results)
        
        return stats_dict
    
    def save_statistics(self, stats_dict: Dict[str, pd.DataFrame]):
        """Save statistics to files."""
        for name, df in stats_dict.items():
            output_file = self.output_dir / 'statistics' / f'{name}_statistics.csv'
            df.to_csv(output_file)
            print(f"Saved {name} statistics to {output_file}")
    
    def generate_visualizations(self, df: pd.DataFrame):
        """Generate all required visualizations."""
        print("Generating visualizations...")
        
        # Set color palette
        colors = {'raw': '#E74C3C', 'preproc': '#3498DB', 'difference': '#2ECC71'}
        
        # 1. Before vs After Boxplots
        self.plot_before_after_boxplots(df, colors)
        
        # 2. Before vs After Violin Plots
        self.plot_before_after_violin(df, colors)
        
        # 3. Histograms
        self.plot_histograms(df, colors)
        
        # 4. Density Plots
        self.plot_density(df, colors)
        
        # 5. Scatter Plots
        self.plot_scatter(df, colors)
        
        # 6. Bland-Altman Plots
        self.plot_bland_altman(df, colors)
        
        # 7. Correlation Heatmaps
        self.plot_correlation_heatmap(df)
        
        # 8. Subject-wise Improvement Plots
        self.plot_subject_improvement(df, colors)
        
        # 9. Image Similarity Distributions
        self.plot_similarity_distributions(df, colors)
        
        # 10. Motion Summary
        self.plot_motion_summary(df, colors)
        
        print("Visualization generation complete!")
    
    def save_figure(self, fig, name: str):
        """Save figure in multiple formats and resolutions."""
        # PNG 300 DPI
        fig.savefig(self.output_dir / 'plots' / 'dpi300' / f'{name}.png', dpi=300)
        
        # PNG 600 DPI
        fig.savefig(self.output_dir / 'plots' / 'dpi600' / f'{name}.png', dpi=600)
        
        # PDF
        fig.savefig(self.output_dir / 'plots' / 'pdf' / f'{name}.pdf', format='pdf')
        
        # SVG
        fig.savefig(self.output_dir / 'plots' / 'svg' / f'{name}.svg', format='svg')
        
        plt.close(fig)
    
    def plot_before_after_boxplots(self, df: pd.DataFrame, colors: dict):
        """Generate before vs after boxplots."""
        metrics = ['snr', 'tsnr', 'mean_signal', 'global_mean']
        
        for metric in metrics:
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col not in df.columns or preproc_col not in df.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data_to_plot = [df[raw_col].dropna(), df[preproc_col].dropna()]
            bp = ax.boxplot(data_to_plot, labels=['Raw', 'Preprocessed'], patch_artist=True)
            
            bp['boxes'][0].set_facecolor(colors['raw'])
            bp['boxes'][1].set_facecolor(colors['preproc'])
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Before vs After: {metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            self.save_figure(fig, f'{metric}_boxplot')
    
    def plot_before_after_violin(self, df: pd.DataFrame, colors: dict):
        """Generate before vs after violin plots."""
        metrics = ['snr', 'tsnr', 'mean_signal', 'global_mean']
        
        for metric in metrics:
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col not in df.columns or preproc_col not in df.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data for seaborn
            plot_df = pd.DataFrame({
                'Condition': ['Raw'] * len(df[raw_col].dropna()) + ['Preprocessed'] * len(df[preproc_col].dropna()),
                metric.replace('_', ' ').title(): pd.concat([df[raw_col].dropna(), df[preproc_col].dropna()])
            })
            
            sns.violinplot(data=plot_df, x='Condition', y=metric.replace('_', ' ').title(),
                          palette=[colors['raw'], colors['preproc']], ax=ax)
            
            ax.set_title(f'Before vs After: {metric.replace("_", " ").title()} Distribution')
            ax.grid(True, alpha=0.3)
            
            self.save_figure(fig, f'{metric}_violin')
    
    def plot_histograms(self, df: pd.DataFrame, colors: dict):
        """Generate histograms for key metrics."""
        # Use metrics that have actual values (psnr/ssim are NaN for different spaces)
        metrics = ['snr', 'tsnr', 'mean_signal', 'global_std']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col in df.columns and preproc_col in df.columns:
                ax.hist(df[raw_col].dropna(), bins=20, color=colors['raw'], alpha=0.5, 
                       label='Raw', edgecolor='black')
                ax.hist(df[preproc_col].dropna(), bins=20, color=colors['preproc'], alpha=0.5,
                       label='Preprocessed', edgecolor='black')
                ax.set_xlabel(metric.replace('_', ' ').upper())
                ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
                ax.legend()
            
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'metrics_histograms')
    
    def plot_density(self, df: pd.DataFrame, colors: dict):
        """Generate KDE density plots."""
        metrics = ['snr', 'tsnr', 'mean_signal']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col in df.columns and preproc_col in df.columns:
                raw_data = df[raw_col].dropna()
                preproc_data = df[preproc_col].dropna()
                
                if len(raw_data) > 1:
                    sns.kdeplot(data=raw_data, ax=ax, color=colors['raw'], label='Raw', linewidth=2)
                if len(preproc_data) > 1:
                    sns.kdeplot(data=preproc_data, ax=ax, color=colors['preproc'], label='Preprocessed', linewidth=2)
                
                ax.set_xlabel(metric.replace('_', ' ').title())
                ax.set_ylabel('Density')
                ax.set_title(f'{metric.replace("_", " ").title()} Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'metrics_density')
    
    def plot_scatter(self, df: pd.DataFrame, colors: dict):
        """Generate scatter plots of raw vs preprocessed."""
        metrics = ['snr', 'tsnr', 'mean_signal', 'global_mean']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col in df.columns and preproc_col in df.columns:
                valid = df[[raw_col, preproc_col]].dropna()
                
                if len(valid) > 0:
                    ax.scatter(valid[raw_col], valid[preproc_col], alpha=0.6, 
                              c=colors['difference'], s=50, edgecolor='black')
                    
                    # Add identity line
                    min_val = min(valid[raw_col].min(), valid[preproc_col].min())
                    max_val = max(valid[raw_col].max(), valid[preproc_col].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Identity')
                    
                    # Add regression line
                    z = np.polyfit(valid[raw_col], valid[preproc_col], 1)
                    p = np.poly1d(z)
                    ax.plot(valid[raw_col].sort_values(), p(valid[raw_col].sort_values()), 
                           'r-', linewidth=2, label='Regression')
                    
                    ax.set_xlabel(f'Raw {metric.replace("_", " ").title()}')
                    ax.set_ylabel(f'Preprocessed {metric.replace("_", " ").title()}')
                    ax.set_title(f'Raw vs Preprocessed: {metric.replace("_", " ").title()}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'raw_vs_preproc_scatter')
    
    def plot_bland_altman(self, df: pd.DataFrame, colors: dict):
        """Generate Bland-Altman plots."""
        metrics = ['snr', 'tsnr', 'mean_signal', 'global_mean']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col in df.columns and preproc_col in df.columns:
                valid = df[[raw_col, preproc_col]].dropna()
                
                if len(valid) > 0:
                    mean_vals = (valid[raw_col] + valid[preproc_col]) / 2
                    diff_vals = valid[preproc_col] - valid[raw_col]
                    
                    ax.scatter(mean_vals, diff_vals, alpha=0.6, c=colors['difference'], 
                              s=50, edgecolor='black')
                    
                    # Add mean and limits of agreement
                    mean_diff = np.mean(diff_vals)
                    std_diff = np.std(diff_vals)
                    
                    ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label='Mean')
                    ax.axhline(mean_diff + 1.96*std_diff, color='blue', linestyle='--', 
                              linewidth=2, label='+1.96 SD')
                    ax.axhline(mean_diff - 1.96*std_diff, color='blue', linestyle='--', 
                              linewidth=2, label='-1.96 SD')
                    
                    ax.set_xlabel(f'Mean of Raw and Preprocessed {metric.replace("_", " ").title()}')
                    ax.set_ylabel(f'Difference (Preprocessed - Raw)')
                    ax.set_title(f'Bland-Altman: {metric.replace("_", " ").title()}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'bland_altman_plots')
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Generate correlation heatmap of all metrics."""
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove columns with all NaN
        numeric_df = numeric_df.dropna(axis=1, how='all')
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 14))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, ax=ax,
                   cbar_kws={'shrink': 0.8})
        
        ax.set_title('Correlation Matrix of Quality Metrics')
        
        plt.tight_layout()
        self.save_figure(fig, 'correlation_heatmap')
    
    def plot_subject_improvement(self, df: pd.DataFrame, colors: dict):
        """Generate subject-wise improvement plots."""
        metrics = ['snr', 'tsnr']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            raw_col = f'raw_{metric}'
            preproc_col = f'preproc_{metric}'
            
            if raw_col in df.columns and preproc_col in df.columns:
                valid = df[['subject', raw_col, preproc_col]].dropna()
                
                if len(valid) > 0:
                    valid['improvement'] = valid[preproc_col] - valid[raw_col]
                    valid = valid.sort_values('improvement')
                    
                    colors_bar = [colors['preproc'] if x > 0 else colors['raw'] 
                                 for x in valid['improvement']]
                    
                    ax.barh(range(len(valid)), valid['improvement'], color=colors_bar, alpha=0.7)
                    ax.set_yticks(range(len(valid)))
                    ax.set_yticklabels(valid['subject'], fontsize=8)
                    ax.set_xlabel(f'{metric.upper()} Change (Preprocessed - Raw)')
                    ax.set_title(f'Subject-wise {metric.upper()} Change')
                    ax.axvline(0, color='black', linestyle='-', linewidth=1)
                    ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        self.save_figure(fig, 'subject_improvement')
    
    def plot_similarity_distributions(self, df: pd.DataFrame, colors: dict):
        """Generate distributions of image similarity metrics."""
        # Use distribution-based metrics (voxel-wise metrics are NaN for different spaces)
        metrics = ['ks_statistic', 'wasserstein_distance', 'bhattacharyya_coeff', 
                   'mean_diff_cohen_d', 'raw_cv', 'preproc_cv']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            if metric in df.columns:
                data = df[metric].dropna()
                
                if len(data) > 0:
                    ax.hist(data, bins=20, color=colors['difference'], alpha=0.7, edgecolor='black')
                    ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {np.mean(data):.3f}')
                    ax.set_xlabel(metric.replace('_', ' ').upper())
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'similarity_distributions')
    
    def plot_motion_summary(self, df: pd.DataFrame, colors: dict):
        """Generate motion metrics summary plot."""
        motion_metrics = ['mean_fd', 'max_fd', 'std_fd', 'pct_fd_gt_02', 'pct_fd_gt_05']
        
        available_metrics = [m for m in motion_metrics if m in df.columns]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(available_metrics[:6]):
            ax = axes[idx]
            
            if metric in df.columns:
                data = df[metric].dropna()
                
                if len(data) > 0:
                    ax.bar(range(len(df)), df[metric].fillna(0), color=colors['preproc'], alpha=0.7)
                    ax.set_xlabel('Subject')
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.set_title(f'{metric.replace("_", " ").title()} by Subject')
                    ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(available_metrics), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        self.save_figure(fig, 'motion_summary')


def main():
    """Main function to run the quality metrics analysis."""
    # Configuration
    raw_dir = '/root/fMRI/ds004302-download'
    preproc_dir = '/root/fMRI/output'
    output_dir = '/root/fMRI/quality_analysis'
    
    # Initialize framework
    print("=" * 60)
    print("fMRI Quality Metrics Framework")
    print("=" * 60)
    print(f"Raw data directory: {raw_dir}")
    print(f"Preprocessed data directory: {preproc_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    framework = FMRIQualityMetrics(raw_dir, preproc_dir, output_dir, n_jobs=-1)
    
    # Run analysis
    df = framework.run_analysis()
    
    # Compute statistics
    stats_dict = framework.compute_statistics(df)
    framework.save_statistics(stats_dict)
    
    # Generate visualizations
    framework.generate_visualizations(df)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"  - Metrics: {output_dir}/metrics/")
    print(f"  - Statistics: {output_dir}/statistics/")
    print(f"  - Plots: {output_dir}/plots/")
    print("=" * 60)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    
    if 'raw_snr' in df.columns and 'preproc_snr' in df.columns:
        print(f"SNR - Raw: {df['raw_snr'].mean():.3f} ± {df['raw_snr'].std():.3f}")
        print(f"SNR - Preprocessed: {df['preproc_snr'].mean():.3f} ± {df['preproc_snr'].std():.3f}")
    
    if 'raw_tsnr' in df.columns and 'preproc_tsnr' in df.columns:
        print(f"tSNR - Raw: {df['raw_tsnr'].mean():.3f} ± {df['raw_tsnr'].std():.3f}")
        print(f"tSNR - Preprocessed: {df['preproc_tsnr'].mean():.3f} ± {df['preproc_tsnr'].std():.3f}")
    
    if 'psnr' in df.columns:
        print(f"PSNR: {df['psnr'].mean():.3f} ± {df['psnr'].std():.3f}")
    
    if 'ssim' in df.columns:
        print(f"SSIM: {df['ssim'].mean():.3f} ± {df['ssim'].std():.3f}")
    
    print("-" * 60)


if __name__ == '__main__':
    main()
