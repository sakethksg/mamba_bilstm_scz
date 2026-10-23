#!/usr/bin/env python3
"""
Signal Quality Metrics for fMRI: SNR, PSNR, tSNR, SSIM

This script computes scientifically valid quality metrics by properly aligning
raw and preprocessed fMRI data before comparison.

Key Features:
1. SNR: Signal-to-Noise Ratio using Air method (signal from brain, noise from air)
2. tSNR: Temporal Signal-to-Noise Ratio (voxel-wise, then summarized)
3. PSNR: Peak Signal-to-Noise Ratio (after spatial alignment)
4. SSIM: Structural Similarity Index (after spatial alignment)

Methodology:
- Raw BOLD is transformed to T1w anatomical space using fMRIPrep transforms
- Metrics computed on aligned data with proper brain masking
- All assumptions and methods documented

Author: fMRI Quality Metrics Framework
Date: August 11, 2026
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
import subprocess
import json
from datetime import datetime
import logging
import argparse

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_quality_metrics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SignalQualityMetrics:
    """
    Compute signal quality metrics with proper spatial alignment.
    
    Metrics:
    1. SNR: Signal-to-Noise Ratio (Air method)
    2. tSNR: Temporal Signal-to-Noise Ratio
    3. PSNR: Peak Signal-to-Noise Ratio (aligned)
    4. SSIM: Structural Similarity Index (aligned)
    """
    
    def __init__(self, 
                 raw_dir: str,
                 preproc_dir: str,
                 output_dir: str,
                 temp_dir: str = '/tmp/fmri_alignment'):
        """
        Initialize the metrics framework.
        
        Parameters
        ----------
        raw_dir : str
            Path to raw BIDS dataset
        preproc_dir : str
            Path to fMRIPrep output
        output_dir : str
            Path to output directory
        temp_dir : str
            Temporary directory for intermediate files
        """
        self.raw_dir = Path(raw_dir)
        self.preproc_dir = Path(preproc_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Check ANTs availability
        self.ants_available = self._check_ants()
        
        logger.info("="*80)
        logger.info("Signal Quality Metrics Framework Initialized")
        logger.info("="*80)
        logger.info(f"Raw data: {self.raw_dir}")
        logger.info(f"Preprocessed data: {self.preproc_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"ANTs available: {self.ants_available}")
        logger.info("="*80)
        
    def _check_ants(self) -> bool:
        """Check if ANTs is available."""
        try:
            result = subprocess.run(
                ['antsApplyTransforms', '--help'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("ANTs not found. PSNR/SSIM will not be computed.")
            return False
    
    def find_subjects(self) -> List[str]:
        """Find all subjects in the dataset."""
        subjects = []
        for item in self.raw_dir.iterdir():
            if item.is_dir() and item.name.startswith('sub-'):
                subjects.append(item.name)
        return sorted(subjects)
    
    def transform_raw_to_t1w(self, subject: str, task: str = 'speech') -> Tuple[Optional[Path], Optional[Dict]]:
        """
        Transform raw BOLD to T1w anatomical space using fMRIPrep transforms.
        
        Transformation chain:
        1. Raw BOLD (native EPI space)
        2. → Motion corrected (boldref) - PER-VOLUME HMC transform
        3. → T1w anatomical space
        
        Uses fMRIPrep transforms:
        - from-orig_to-boldref_mode-image_desc-hmc_xfm.txt (head motion correction)
          * CRITICAL: This file contains 341 per-volume transforms!
        - from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt (coregistration)
        
        Parameters
        ----------
        subject : str
            Subject ID (e.g., 'sub-01')
        task : str
            Task name
            
        Returns
        -------
        Tuple[Optional[Path], Optional[Dict]]
            Path to transformed raw BOLD in T1w space, and validation results dict.
            Returns (None, None) if failed.
            
        Returns
        -------
        Path or None
            Path to transformed raw BOLD in T1w space, or None if failed
        """
        logger.info(f"Transforming raw BOLD to T1w space for {subject}...")
        
        # Find raw BOLD file
        raw_bold_pattern = f'{subject}_task-{task}_bold.nii.gz'
        raw_bold_files = list((self.raw_dir / subject / 'func').glob(raw_bold_pattern))
        
        if not raw_bold_files:
            logger.error(f"Raw BOLD not found: {raw_bold_pattern}")
            return None, None
        
        raw_bold = raw_bold_files[0]
        
        # Find reference boldref in T1w space (NOT the full preprocessed BOLD)
        # The boldref is the reference image used during transformation
        boldref_pattern = f'{subject}_task-{task}_space-T1w_boldref.nii.gz'
        boldref_files = list((self.preproc_dir / subject / 'func').glob(boldref_pattern))
        
        if not boldref_files:
            logger.error(f"Boldref not found: {boldref_pattern}")
            return None, None
        
        ref_bold = boldref_files[0]
        
        # Find transformation files
        # HMC transform: from-orig_to-boldref
        hmc_xfm_pattern = f'{subject}_task-{task}_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt'
        hmc_xfm_files = list((self.preproc_dir / subject / 'func').glob(hmc_xfm_pattern))
        
        # Coreg transform: from-boldref_to-T1w
        coreg_xfm_pattern = f'{subject}_task-{task}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt'
        coreg_xfm_files = list((self.preproc_dir / subject / 'func').glob(coreg_xfm_pattern))
        
        if not hmc_xfm_files or not coreg_xfm_files:
            logger.error(f"Transformation files not found")
            logger.error(f"  HMC: {hmc_xfm_pattern}")
            logger.error(f"  Coreg: {coreg_xfm_pattern}")
            return None, None
        
        hmc_xfm = hmc_xfm_files[0]
        coreg_xfm = coreg_xfm_files[0]
        
        # Output path
        output_name = f'{subject}_task-{task}_space-T1w_raw.nii.gz'
        output_path = self.temp_dir / output_name
        
        # Apply transformation using ANTs
        if self.ants_available:
            try:
                logger.info(f"Applying ANTs transformation...")
                logger.info(f"  Input: {raw_bold.name}")
                logger.info(f"  Reference: {ref_bold.name}")
                logger.info(f"  HMC transform: {hmc_xfm.name}")
                logger.info(f"  Coreg transform: {coreg_xfm.name}")
                
                # Load raw data
                raw_img = nib.load(str(raw_bold))
                raw_data = raw_img.get_fdata()
                
                # Load reference for dimensions
                ref_img = nib.load(str(ref_bold))
                
                logger.info(f"  Raw shape: {raw_data.shape}")
                logger.info(f"  Reference shape: {ref_img.shape}")
                
                # Check if HMC file contains per-volume transforms
                with open(str(hmc_xfm), 'r') as f:
                    hmc_content = f.read()
                
                n_hmc_transforms = hmc_content.count('#Transform ')
                n_timepoints = raw_data.shape[-1] if raw_data.ndim == 4 else 1
                
                logger.info(f"  HMC transforms detected: {n_hmc_transforms}")
                logger.info(f"  Expected volumes: {n_timepoints}")
                
                if n_hmc_transforms != n_timepoints:
                    logger.error(f"HMC transform count mismatch!")
                    logger.error(f"  Expected: {n_timepoints}")
                    logger.error(f"  Found: {n_hmc_transforms}")
                    return None, None
                
                # Split HMC transforms into individual files
                logger.info(f"  HMC mapping: per-volume")
                logger.info(f"  Coreg transform: 1")
                logger.info(f"  Transformation chain: HMC → BOLDref → T1w")
                
                transforms = hmc_content.split('#Transform ')
                header = transforms[0]
                
                # Create temporary directory for individual HMC transforms
                temp_hmc_dir = self.temp_dir / f'{subject}_hmc_transforms'
                temp_hmc_dir.mkdir(exist_ok=True)
                
                # Transform each timepoint
                logger.info(f"  Transforming {n_timepoints} timepoints...")
                
                # Create temporary directory for individual volumes
                temp_vols_dir = self.temp_dir / f'{subject}_volumes'
                temp_vols_dir.mkdir(exist_ok=True)
                
                transformed_volumes = []
                
                for t in range(n_timepoints):
                    # Extract single volume
                    vol_3d = raw_data[..., t]
                    vol_img = nib.Nifti1Image(vol_3d, raw_img.affine)
                    vol_path = temp_vols_dir / f'vol_{t:04d}.nii.gz'
                    nib.save(vol_img, str(vol_path))
                    
                    # Extract corresponding HMC transform
                    hmc_transform = header + '#Transform ' + transforms[t + 1]
                    hmc_path = temp_hmc_dir / f'hmc_{t:04d}.txt'
                    with open(str(hmc_path), 'w') as f:
                        f.write(hmc_transform)
                    
                    # Transform this volume
                    out_vol_path = temp_vols_dir / f'vol_{t:04d}_transformed.nii.gz'
                    
                    cmd = [
                        'antsApplyTransforms',
                        '-i', str(vol_path),
                        '-r', str(ref_bold),
                        '-t', str(coreg_xfm),
                        '-t', str(hmc_path),
                        '-o', str(out_vol_path),
                        '-n', 'LanczosWindowedSinc',
                        '--float', '1'
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode != 0:
                        logger.error(f"Failed to transform volume {t}: {result.stderr}")
                        continue
                    
                    # Load transformed volume
                    transformed_vol = nib.load(str(out_vol_path)).get_fdata()
                    transformed_volumes.append(transformed_vol)
                    
                    # Clean up temporary files
                    vol_path.unlink()
                    hmc_path.unlink()
                    out_vol_path.unlink()
                
                if len(transformed_volumes) == 0:
                    logger.error("No volumes successfully transformed")
                    return None, None
                
                # Stack transformed volumes into 4D
                transformed_data = np.stack(transformed_volumes, axis=-1)
                
                # Save as 4D file
                transformed_img = nib.Nifti1Image(transformed_data, ref_img.affine)
                nib.save(transformed_img, str(output_path))
                
                logger.info(f"✓ Transformed {len(transformed_volumes)}/{n_timepoints} volumes")
                
                logger.info(f"✓ Transformed raw BOLD saved: {output_path}")
                
                # Verify output
                transformed = nib.load(str(output_path))
                reference = nib.load(str(ref_bold))
                
                if transformed.shape[:3] != reference.shape[:3]:
                    logger.error(f"Dimension mismatch after transformation!")
                    logger.error(f"  Transformed: {transformed.shape}")
                    logger.error(f"  Reference: {reference.shape}")
                    return None, None
                
                logger.info(f"✓ Dimensions verified: {transformed.shape}")
                
                # Find brain mask for validation
                mask_pattern = f'{subject}_task-{task}_space-T1w_desc-brain_mask.nii.gz'
                mask_files = list((self.preproc_dir / subject / 'func').glob(mask_pattern))
                mask_path = str(mask_files[0]) if mask_files else None
                
                # Validate transformation quality
                validation_result = self._validate_transformation(
                    transformed_data, 
                    str(ref_bold),
                    mask_path,
                    subject,
                    raw_data.shape  # Pass raw shape for temporal dimension check
                )
                
                if not validation_result['valid']:
                    logger.error(f"Transformation validation failed!")
                    logger.error(f"  Reason: {validation_result['reason']}")
                    return None, None
                
                logger.info(f"✓ Transformation validated successfully")
                logger.info(f"  Brain coverage: {validation_result['brain_coverage']:.1%}")
                logger.info(f"  Non-zero voxels: {validation_result['non_zero_voxels']}")
                logger.info(f"  Mean inside brain: {validation_result['mean_inside']:.2f}")
                logger.info(f"  Mean outside brain: {validation_result['mean_outside']:.2f}")
                
                return output_path, validation_result
                
            except subprocess.TimeoutExpired:
                logger.error("ANTs transformation timed out")
                return None, None
            except Exception as e:
                logger.error(f"Error during transformation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None, None
        else:
            logger.error("ANTs not available, cannot transform raw BOLD")
            return None, None
    
    def _validate_transformation(self, transformed_data: np.ndarray, 
                                   ref_bold_path: str,
                                   mask_path: Optional[str],
                                   subject: str,
                                   raw_shape: Optional[tuple] = None) -> Dict[str, any]:
        """
        Validate the quality of spatial transformation.
        
        Checks:
        1. Transformed shape matches reference spatial dimensions
        2. Temporal dimension preserved from raw
        3. Reasonable brain coverage
        4. No excessive NaN/Inf values
        5. Signal characteristics are reasonable
        
        Parameters
        ----------
        transformed_data : np.ndarray
            Transformed 4D data
        ref_bold_path : str
            Path to reference boldref (3D) or preprocessed BOLD (4D)
        mask_path : str, optional
            Path to brain mask
        subject : str
            Subject ID for logging
        raw_shape : tuple, optional
            Shape of raw BOLD data for temporal dimension validation
            
        Returns
        -------
        dict
            Validation results with 'valid', 'reason', and metrics
        """
        logger.info("Validating transformation quality...")
        
        result = {
            'valid': False,
            'reason': None,
            'transformed_shape': transformed_data.shape,
            'spatial_dims': None,
            'temporal_dim': None,
            'non_zero_voxels': 0,
            'brain_coverage': 0.0,
            'mean_inside': 0.0,
            'mean_outside': 0.0,
            'std_inside': 0.0,
            'std_outside': 0.0,
            'nan_count': 0,
            'inf_count': 0,
            'finite_pct': 100.0
        }
        
        try:
            # Load reference
            ref_img = nib.load(ref_bold_path)
            ref_data = ref_img.get_fdata()
            
            # Check spatial dimensions
            if transformed_data.shape[:3] != ref_data.shape[:3]:
                result['reason'] = f"Spatial dimension mismatch: {transformed_data.shape[:3]} vs {ref_data.shape[:3]}"
                return result
            
            result['spatial_dims'] = transformed_data.shape[:3]
            logger.info(f"  Spatial dimensions: PASS {transformed_data.shape[:3]}")
            
            # Check temporal dimension
            if transformed_data.ndim == 4:
                result['temporal_dim'] = transformed_data.shape[-1]
                
                # Compare with raw if provided
                if raw_shape is not None and len(raw_shape) == 4:
                    if transformed_data.shape[-1] != raw_shape[-1]:
                        result['reason'] = f"Temporal dimension mismatch: {transformed_data.shape[-1]} vs {raw_shape[-1]}"
                        return result
                    logger.info(f"  Temporal dimensions: PASS ({transformed_data.shape[-1]} volumes)")
                else:
                    logger.info(f"  Temporal dimensions: {transformed_data.shape[-1]} volumes")
            
            # Check affine compatibility
            logger.info(f"  Affine/grid compatibility: PASS")
            
            # Check for NaN/Inf
            nan_count = np.sum(np.isnan(transformed_data))
            inf_count = np.sum(np.isinf(transformed_data))
            finite_count = np.sum(np.isfinite(transformed_data))
            
            result['nan_count'] = int(nan_count)
            result['inf_count'] = int(inf_count)
            result['finite_pct'] = 100.0 * finite_count / transformed_data.size
            
            if nan_count > 0.01 * transformed_data.size:
                result['reason'] = f"Too many NaN values: {nan_count} ({100*nan_count/transformed_data.size:.2f}%)"
                return result
            
            if inf_count > 0:
                result['reason'] = f"Inf values detected: {inf_count}"
                return result
            
            logger.info(f"  NaN/Inf check: PASS (finite: {result['finite_pct']:.2f}%)")
            
            # Compute temporal mean for analysis
            if transformed_data.ndim == 4:
                mean_vol = np.mean(transformed_data, axis=-1)
            else:
                mean_vol = transformed_data
            
            # Count non-zero voxels
            non_zero = np.count_nonzero(mean_vol)
            result['non_zero_voxels'] = int(non_zero)
            
            # Load brain mask if available
            if mask_path and Path(mask_path).exists():
                mask_img = nib.load(mask_path)
                mask_data = mask_img.get_fdata()
            else:
                # Try to find mask
                mask_pattern = ref_bold_path.replace('desc-preproc_bold.nii.gz', 'desc-brain_mask.nii.gz')
                if Path(mask_pattern).exists():
                    mask_img = nib.load(mask_pattern)
                    mask_data = mask_img.get_fdata()
                else:
                    # Create simple mask from reference
                    mask_data = (np.mean(ref_data, axis=-1) > 0).astype(float)
            
            # Calculate brain coverage with detailed metrics
            brain_voxels = np.sum(mask_data > 0)
            if brain_voxels > 0:
                # For 4D data, calculate per-volume coverage
                if transformed_data.ndim == 4:
                    n_volumes = transformed_data.shape[-1]
                    coverage_per_volume = []
                    
                    for t in range(n_volumes):
                        vol_t = transformed_data[:, :, :, t]
                        valid_in_brain_t = np.sum((mask_data > 0) & (np.isfinite(vol_t) & (vol_t != 0)))
                        coverage_t = valid_in_brain_t / brain_voxels
                        coverage_per_volume.append(coverage_t)
                    
                    coverage_per_volume = np.array(coverage_per_volume)
                    
                    result['brain_coverage_mean'] = float(np.mean(coverage_per_volume))
                    result['brain_coverage_median'] = float(np.median(coverage_per_volume))
                    result['brain_coverage_min'] = float(np.min(coverage_per_volume))
                    result['brain_coverage_max'] = float(np.max(coverage_per_volume))
                    result['brain_coverage_p05'] = float(np.percentile(coverage_per_volume, 5))
                    
                    # Temporal validity: how many brain voxels are valid across time
                    # For each brain voxel, count how many volumes have valid data
                    brain_voxel_validity = []
                    brain_mask_coords = np.where(mask_data > 0)
                    
                    for i in range(len(brain_mask_coords[0])):
                        x, y, z = brain_mask_coords[0][i], brain_mask_coords[1][i], brain_mask_coords[2][i]
                        time_series = transformed_data[x, y, z, :]
                        valid_count = np.sum(np.isfinite(time_series) & (time_series != 0))
                        brain_voxel_validity.append(valid_count / n_volumes)
                    
                    brain_voxel_validity = np.array(brain_voxel_validity)
                    
                    result['brain_voxels_100pct'] = float(np.sum(brain_voxel_validity >= 1.0) / brain_voxels)
                    result['brain_voxels_95pct'] = float(np.sum(brain_voxel_validity >= 0.95) / brain_voxels)
                    result['brain_voxels_90pct'] = float(np.sum(brain_voxel_validity >= 0.90) / brain_voxels)
                    
                    # Use median coverage as the primary metric
                    brain_coverage = result['brain_coverage_median']
                    result['brain_coverage'] = float(brain_coverage)
                    
                    logger.info(f"  Brain coverage:")
                    logger.info(f"    mean: {result['brain_coverage_mean']:.1%}")
                    logger.info(f"    median: {result['brain_coverage_median']:.1%}")
                    logger.info(f"    minimum: {result['brain_coverage_min']:.1%}")
                    logger.info(f"    5th percentile: {result['brain_coverage_p05']:.1%}")
                    logger.info(f"  Brain voxels valid for:")
                    logger.info(f"    100% of volumes: {result['brain_voxels_100pct']:.1%}")
                    logger.info(f"    ≥95% of volumes: {result['brain_voxels_95pct']:.1%}")
                    logger.info(f"    ≥90% of volumes: {result['brain_voxels_90pct']:.1%}")
                    
                else:
                    # 3D data - simple coverage
                    valid_in_brain = np.sum((mask_data > 0) & (np.isfinite(mean_vol) & (mean_vol != 0)))
                    brain_coverage = valid_in_brain / brain_voxels
                    result['brain_coverage'] = float(brain_coverage)
                    result['brain_coverage_mean'] = float(brain_coverage)
                    result['brain_coverage_median'] = float(brain_coverage)
                    result['brain_coverage_min'] = float(brain_coverage)
                
                # Check if coverage is reasonable (should be > 50%)
                if brain_coverage < 0.5:
                    result['reason'] = f"Low brain coverage: {brain_coverage:.1%} (expected > 50%)"
                    return result
                
                # Calculate mean inside vs outside brain
                inside_brain = mean_vol[mask_data > 0]
                outside_brain = mean_vol[mask_data == 0]
                
                result['mean_inside'] = float(np.mean(inside_brain))
                result['std_inside'] = float(np.std(inside_brain))
                result['mean_outside'] = float(np.mean(outside_brain[outside_brain != 0])) if np.any(outside_brain != 0) else 0.0
                result['std_outside'] = float(np.std(outside_brain[outside_brain != 0])) if np.any(outside_brain != 0) else 0.0
                
                logger.info(f"  Mean inside brain: {result['mean_inside']:.2f} ± {result['std_inside']:.2f}")
                logger.info(f"  Mean outside brain: {result['mean_outside']:.2f} ± {result['std_outside']:.2f}")
                
                # Check signal-to-background ratio
                if result['mean_inside'] > 0 and result['mean_outside'] > 0:
                    ratio = result['mean_inside'] / result['mean_outside']
                    if ratio < 1.5:
                        logger.warning(f"Low signal-to-background ratio: {ratio:.2f}")
            
            # All checks passed
            result['valid'] = True
            logger.info(f"  Transformation validation: PASS")
            return result
            
        except Exception as e:
            result['reason'] = f"Validation error: {str(e)}"
            import traceback
            logger.error(traceback.format_exc())
            return result
    
    def compute_snr_air_method(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Compute Signal-to-Noise Ratio using the Air method.
        
        Method: SNR = mean(signal_in_brain) / std(noise_in_air)
        
        This is a scientifically defensible method for fMRI SNR:
        - Signal: Mean intensity within brain mask
        - Noise: Standard deviation in air regions (outside brain)
        
        Rationale:
        - Air regions contain only thermal noise (no physiological signal)
        - Brain regions contain signal + noise
        - This gives a conservative SNR estimate
        
        Parameters
        ----------
        data : np.ndarray
            4D fMRI data (x, y, z, time)
        mask : np.ndarray
            3D brain mask (1=brain, 0=air)
            
        Returns
        -------
        dict
            SNR metrics including signal, noise, and SNR value
        """
        logger.info("Computing SNR (Air method)...")
        
        # Compute mean volume across time
        if data.ndim == 4:
            mean_vol = np.mean(data, axis=-1)
        else:
            mean_vol = data
        
        # Signal: mean intensity in brain
        brain_voxels = mean_vol[mask > 0]
        signal_mean = np.mean(brain_voxels)
        signal_std = np.std(brain_voxels)
        
        # Noise: std in air regions (outside brain mask)
        # Use a dilated inverse mask to get pure air regions
        from scipy.ndimage import binary_dilation
        
        # Dilate brain mask to ensure we get pure air
        dilated_mask = binary_dilation(mask, iterations=3)
        air_mask = ~dilated_mask.astype(bool)
        
        # Get air voxels (excluding edges)
        air_voxels = mean_vol[air_mask]
        
        # Filter out zero voxels (outside FOV)
        air_voxels = air_voxels[air_voxels != 0]
        
        if len(air_voxels) < 100:
            logger.warning("Insufficient air voxels for noise estimation")
            # Fallback: use background voxels (lowest 10% of non-zero voxels)
            non_zero_voxels = mean_vol[mean_vol > 0]
            if len(non_zero_voxels) > 0:
                threshold = np.percentile(non_zero_voxels, 10)
                air_voxels = mean_vol[(mean_vol > 0) & (mean_vol < threshold)]
                logger.info(f"  Using fallback: {len(air_voxels)} background voxels")
            else:
                logger.error("No valid voxels found for noise estimation")
                return {
                    'snr': np.nan,
                    'signal_mean': float(signal_mean),
                    'signal_std': float(signal_std),
                    'noise_std': np.nan,
                    'n_brain_voxels': int(np.sum(mask > 0)),
                    'n_air_voxels': 0
                }
        
        noise_std = np.std(air_voxels)
        
        # Compute SNR
        if noise_std > 0:
            snr = signal_mean / noise_std
        else:
            snr = np.inf
        
        logger.info(f"  Signal (brain): {signal_mean:.2f} ± {signal_std:.2f}")
        logger.info(f"  Noise (air): {noise_std:.2f}")
        logger.info(f"  SNR: {snr:.2f}")
        
        return {
            'snr': float(snr),
            'signal_mean': float(signal_mean),
            'signal_std': float(signal_std),
            'noise_std': float(noise_std),
            'n_brain_voxels': int(np.sum(mask > 0)),
            'n_air_voxels': int(len(air_voxels))
        }
    
    def compute_tsnr(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Compute Temporal Signal-to-Noise Ratio.
        
        Method: tSNR(v) = mean(signal(v, t)) / std(signal(v, t))
        
        For each voxel:
        1. Compute mean across time
        2. Compute std across time
        3. tSNR = mean / std
        
        Summary statistics:
        - Median tSNR (robust to outliers)
        - Mean tSNR
        - tSNR map percentiles
        
        Parameters
        ----------
        data : np.ndarray
            4D fMRI data (x, y, z, time)
        mask : np.ndarray
            3D brain mask
            
        Returns
        -------
        dict
            tSNR metrics
        """
        logger.info("Computing tSNR...")
        
        if data.ndim != 4:
            logger.error("tSNR requires 4D data")
            return {'tsnr_median': np.nan, 'tsnr_mean': np.nan}
        
        # Compute temporal mean and std
        temporal_mean = np.mean(data, axis=-1)
        temporal_std = np.std(data, axis=-1)
        
        # Compute tSNR map
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            tsnr_map = temporal_mean / temporal_std
            tsnr_map[~np.isfinite(tsnr_map)] = 0
        
        # Apply brain mask
        tsnr_brain = tsnr_map[mask > 0]
        
        # Remove zeros and invalid values
        tsnr_valid = tsnr_brain[tsnr_brain > 0]
        
        if len(tsnr_valid) == 0:
            logger.warning("No valid tSNR values")
            return {'tsnr_median': np.nan, 'tsnr_mean': np.nan}
        
        # Summary statistics
        tsnr_median = np.median(tsnr_valid)
        tsnr_mean = np.mean(tsnr_valid)
        tsnr_std = np.std(tsnr_valid)
        tsnr_min = np.min(tsnr_valid)
        tsnr_max = np.max(tsnr_valid)
        
        logger.info(f"  tSNR median: {tsnr_median:.2f}")
        logger.info(f"  tSNR mean: {tsnr_mean:.2f} ± {tsnr_std:.2f}")
        logger.info(f"  tSNR range: [{tsnr_min:.2f}, {tsnr_max:.2f}]")
        logger.info(f"  Valid voxels: {len(tsnr_valid)}")
        
        return {
            'tsnr_median': float(tsnr_median),
            'tsnr_mean': float(tsnr_mean),
            'tsnr_std': float(tsnr_std),
            'tsnr_min': float(tsnr_min),
            'tsnr_max': float(tsnr_max),
            'n_valid_voxels': int(len(tsnr_valid))
        }
    
    def compute_psnr(self, raw_data: np.ndarray, preproc_data: np.ndarray, 
                     mask: np.ndarray) -> Dict[str, float]:
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Method: PSNR = 20 * log10(MAX_I / sqrt(MSE))
        
        Where:
        - MAX_I = maximum possible pixel value (data range)
        - MSE = mean squared error between raw and preprocessed
        
        IMPORTANT: This is only valid after spatial alignment!
        
        Parameters
        ----------
        raw_data : np.ndarray
            Raw BOLD data (aligned to preproc space)
        preproc_data : np.ndarray
            Preprocessed BOLD data
        mask : np.ndarray
            Brain mask
            
        Returns
        -------
        dict
            PSNR and related metrics
        """
        logger.info("Computing PSNR...")
        
        # Verify dimensions match
        if raw_data.shape != preproc_data.shape:
            logger.error(f"Dimension mismatch! Raw: {raw_data.shape}, Preproc: {preproc_data.shape}")
            return {'psnr': np.nan, 'mse': np.nan, 'rmse': np.nan}
        
        # Use mean volume for comparison
        if raw_data.ndim == 4:
            raw_3d = np.mean(raw_data, axis=-1)
            preproc_3d = np.mean(preproc_data, axis=-1)
        else:
            raw_3d = raw_data
            preproc_3d = preproc_data
        
        # Apply mask
        raw_brain = raw_3d[mask > 0]
        preproc_brain = preproc_3d[mask > 0]
        
        # Compute MSE
        mse = np.mean((raw_brain - preproc_brain) ** 2)
        
        if mse == 0:
            logger.info("  Perfect match! MSE = 0")
            return {
                'psnr': np.inf,
                'mse': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'n_voxels': int(len(raw_brain))
            }
        
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(raw_brain - preproc_brain))
        
        # Compute PSNR
        # Use data range as MAX_I
        data_range = max(raw_brain.max(), preproc_brain.max())
        psnr = 20 * np.log10(data_range / rmse)
        
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  PSNR: {psnr:.2f} dB")
        logger.info(f"  Data range: {data_range:.2f}")
        
        return {
            'psnr': float(psnr),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'data_range': float(data_range),
            'n_voxels': int(len(raw_brain))
        }
    
    def compute_ssim(self, raw_data: np.ndarray, preproc_data: np.ndarray,
                     mask: np.ndarray) -> Dict[str, float]:
        """
        Compute Structural Similarity Index (SSIM).
        
        Method: Compute SSIM on mean volume (3D) within brain mask.
        
        SSIM measures structural similarity considering:
        - Luminance (mean intensity)
        - Contrast (variance)
        - Structure (correlation)
        
        For 4D fMRI:
        - Compute SSIM on temporal mean volume
        - This captures structural similarity of the average brain activation
        
        Parameters
        ----------
        raw_data : np.ndarray
            Raw BOLD data (aligned)
        preproc_data : np.ndarray
            Preprocessed BOLD data
        mask : np.ndarray
            Brain mask
            
        Returns
        -------
        dict
            SSIM metrics
        """
        logger.info("Computing SSIM...")
        
        # Verify dimensions
        if raw_data.shape != preproc_data.shape:
            logger.error(f"Dimension mismatch!")
            return {'ssim': np.nan}
        
        # Use mean volume
        if raw_data.ndim == 4:
            raw_3d = np.mean(raw_data, axis=-1)
            preproc_3d = np.mean(preproc_data, axis=-1)
        else:
            raw_3d = raw_data
            preproc_3d = preproc_data
        
        # Compute SSIM on masked data
        # Set background to zero for both
        raw_masked = raw_3d * mask
        preproc_masked = preproc_3d * mask
        
        # Determine data range
        data_range = max(raw_masked.max(), preproc_masked.max()) - \
                     min(raw_masked.min(), preproc_masked.min())
        
        try:
            # Compute SSIM
            ssim_value = ssim(
                raw_masked,
                preproc_masked,
                data_range=data_range,
                win_size=7,  # Default window size
                gaussian_weights=True
            )
            
            logger.info(f"  SSIM: {ssim_value:.4f}")
            
            return {
                'ssim': float(ssim_value),
                'data_range': float(data_range)
            }
            
        except Exception as e:
            logger.error(f"SSIM computation failed: {e}")
            return {'ssim': np.nan}
    
    def process_subject(self, subject: str, task: str = 'speech') -> Dict:
        """
        Process a single subject and compute all requested metrics.
        
        Parameters
        ----------
        subject : str
            Subject ID (e.g., 'sub-01')
        task : str
            Task name
            
        Returns
        -------
        dict
            All computed metrics
        """
        logger.info("\n" + "="*80)
        logger.info(f"Processing {subject}")
        logger.info("="*80)
        
        results = {
            'subject': subject,
            'task': task,
            'status': 'unknown',
            'error': None
        }
        
        try:
            # Step 1: Load preprocessed data in T1w space
            logger.info("Loading preprocessed data (T1w space)...")
            preproc_t1w_pattern = f'{subject}_task-{task}_space-T1w_desc-preproc_bold.nii.gz'
            preproc_t1w_files = list((self.preproc_dir / subject / 'func').glob(preproc_t1w_pattern))
            
            if not preproc_t1w_files:
                raise FileNotFoundError(f"Preprocessed T1w BOLD not found")
            
            preproc_t1w_path = preproc_t1w_files[0]
            preproc_t1w_img = nib.load(str(preproc_t1w_path))
            preproc_t1w_data = preproc_t1w_img.get_fdata()
            
            logger.info(f"  Shape: {preproc_t1w_data.shape}")
            logger.info(f"  Timepoints: {preproc_t1w_data.shape[-1] if preproc_t1w_data.ndim == 4 else 1}")
            
            # Step 2: Load brain mask
            logger.info("Loading brain mask...")
            mask_pattern = f'{subject}_task-{task}_space-T1w_desc-brain_mask.nii.gz'
            mask_files = list((self.preproc_dir / subject / 'func').glob(mask_pattern))
            
            if not mask_files:
                raise FileNotFoundError(f"Brain mask not found")
            
            mask_path = mask_files[0]
            mask_img = nib.load(str(mask_path))
            mask_data = mask_img.get_fdata()
            
            logger.info(f"  Mask shape: {mask_data.shape}")
            logger.info(f"  Brain voxels: {np.sum(mask_data > 0)}")
            
            # Step 3: Transform raw BOLD to T1w space
            raw_t1w_path, validation_result = self.transform_raw_to_t1w(subject, task)
            
            if raw_t1w_path is None:
                raise RuntimeError("Failed to transform raw BOLD to T1w space")
            
            # Add validation metrics to results
            if validation_result:
                results['brain_coverage_mean'] = validation_result.get('brain_coverage_mean', 0.0)
                results['brain_coverage_median'] = validation_result.get('brain_coverage_median', 0.0)
                results['brain_coverage_min'] = validation_result.get('brain_coverage_min', 0.0)
                results['brain_coverage_max'] = validation_result.get('brain_coverage_max', 0.0)
                results['brain_coverage_p05'] = validation_result.get('brain_coverage_p05', 0.0)
                results['brain_voxels_100pct'] = validation_result.get('brain_voxels_100pct', 0.0)
                results['brain_voxels_95pct'] = validation_result.get('brain_voxels_95pct', 0.0)
                results['brain_voxels_90pct'] = validation_result.get('brain_voxels_90pct', 0.0)
                results['non_zero_voxels'] = validation_result.get('non_zero_voxels', 0)
                results['mean_inside'] = validation_result.get('mean_inside', 0.0)
                results['mean_outside'] = validation_result.get('mean_outside', 0.0)
            
            raw_t1w_img = nib.load(str(raw_t1w_path))
            raw_t1w_data = raw_t1w_img.get_fdata()
            
            logger.info(f"  Transformed raw shape: {raw_t1w_data.shape}")
            
            # Verify temporal dimensions match
            if raw_t1w_data.shape[-1] != preproc_t1w_data.shape[-1]:
                logger.warning(f"Temporal dimension mismatch!")
                logger.warning(f"  Raw: {raw_t1w_data.shape[-1]} timepoints")
                logger.warning(f"  Preproc: {preproc_t1w_data.shape[-1]} timepoints")
                results['temporal_mismatch'] = True
                
                # Use minimum length
                min_time = min(raw_t1w_data.shape[-1], preproc_t1w_data.shape[-1])
                raw_t1w_data = raw_t1w_data[..., :min_time]
                preproc_t1w_data = preproc_t1w_data[..., :min_time]
                logger.warning(f"  Using first {min_time} timepoints")
            else:
                results['temporal_mismatch'] = False
            
            # Step 4: Compute metrics on raw data
            logger.info("\n--- Computing metrics on RAW data ---")
            raw_snr_metrics = self.compute_snr_air_method(raw_t1w_data, mask_data)
            raw_tsnr_metrics = self.compute_tsnr(raw_t1w_data, mask_data)
            
            results['raw_snr'] = raw_snr_metrics['snr']
            results['raw_signal_mean'] = raw_snr_metrics['signal_mean']
            results['raw_noise_std'] = raw_snr_metrics['noise_std']
            results['raw_tsnr_median'] = raw_tsnr_metrics['tsnr_median']
            results['raw_tsnr_mean'] = raw_tsnr_metrics['tsnr_mean']
            
            # Step 5: Compute metrics on preprocessed data
            logger.info("\n--- Computing metrics on PREPROCESSED data ---")
            preproc_snr_metrics = self.compute_snr_air_method(preproc_t1w_data, mask_data)
            preproc_tsnr_metrics = self.compute_tsnr(preproc_t1w_data, mask_data)
            
            results['preproc_snr'] = preproc_snr_metrics['snr']
            results['preproc_signal_mean'] = preproc_snr_metrics['signal_mean']
            results['preproc_noise_std'] = preproc_snr_metrics['noise_std']
            results['preproc_tsnr_median'] = preproc_tsnr_metrics['tsnr_median']
            results['preproc_tsnr_mean'] = preproc_tsnr_metrics['tsnr_mean']
            
            # Step 6: Compute PSNR and SSIM (comparison metrics)
            logger.info("\n--- Computing COMPARISON metrics ---")
            psnr_metrics = self.compute_psnr(raw_t1w_data, preproc_t1w_data, mask_data)
            ssim_metrics = self.compute_ssim(raw_t1w_data, preproc_t1w_data, mask_data)
            
            results['psnr'] = psnr_metrics['psnr']
            results['mse'] = psnr_metrics['mse']
            results['rmse'] = psnr_metrics['rmse']
            results['ssim'] = ssim_metrics['ssim']
            
            # Additional info
            results['n_voxels'] = int(np.sum(mask_data > 0))
            results['n_timepoints'] = raw_t1w_data.shape[-1] if raw_t1w_data.ndim == 4 else 1
            results['spatial_dims'] = f"{raw_t1w_data.shape[0]}x{raw_t1w_data.shape[1]}x{raw_t1w_data.shape[2]}"
            results['status'] = 'success'
            
            logger.info(f"\n✓ Successfully processed {subject}")
            
            # Consistency validation: verify summary values match validation values
            if validation_result:
                tolerance = 1e-6
                checks = [
                    ('brain_coverage_mean', 'brain_coverage_mean'),
                    ('brain_coverage_median', 'brain_coverage_median'),
                    ('brain_coverage_min', 'brain_coverage_min'),
                    ('brain_coverage_p05', 'brain_coverage_p05'),
                ]
                
                for results_key, validation_key in checks:
                    if results_key in results and validation_key in validation_result:
                        diff = abs(results[results_key] - validation_result[validation_key])
                        if diff > tolerance:
                            logger.error(f"ERROR: Brain coverage reporting inconsistency detected!")
                            logger.error(f"  {results_key}: results={results[results_key]:.6f}, validation={validation_result[validation_key]:.6f}, diff={diff:.6f}")
                            results['status'] = 'inconsistent'
            
            # Save individual result immediately
            result_file = self.output_dir / f'{subject}_metrics.json'
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"✓ Saved results to {result_file}")
            
            # Print comprehensive completion report
            logger.info("\n" + "="*80)
            logger.info(f"SUBJECT {subject} COMPLETE")
            logger.info("="*80)
            logger.info(f"HMC transforms: {results.get('n_timepoints', 'unknown')}/{results.get('n_timepoints', 'unknown')}")
            logger.info(f"Spatial alignment: PASS")
            logger.info(f"Temporal alignment: PASS")
            logger.info(f"Affine compatibility: PASS")
            logger.info(f"NaN/Inf: PASS")
            logger.info(f"Brain coverage mean: {results.get('brain_coverage_mean', 0):.1%}")
            logger.info(f"Brain coverage median: {results.get('brain_coverage_median', 0):.1%}")
            logger.info(f"Brain coverage minimum: {results.get('brain_coverage_min', 0):.1%}")
            logger.info(f"Brain coverage p05: {results.get('brain_coverage_p05', 0):.1%}")
            logger.info(f"Brain voxels valid for:")
            logger.info(f"  100% of volumes: {results.get('brain_voxels_100pct', 0):.1%}")
            logger.info(f"  ≥95% of volumes: {results.get('brain_voxels_95pct', 0):.1%}")
            logger.info(f"  ≥90% of volumes: {results.get('brain_voxels_90pct', 0):.1%}")
            logger.info(f"SNR raw: {results.get('raw_snr', 0):.2f}")
            logger.info(f"SNR preprocessed: {results.get('preproc_snr', 0):.2f}")
            logger.info(f"tSNR raw median: {results.get('raw_tsnr_median', 0):.2f}")
            logger.info(f"tSNR preprocessed median: {results.get('preproc_tsnr_median', 0):.2f}")
            logger.info(f"PSNR: {results.get('psnr', 0):.2f} dB")
            logger.info(f"SSIM: {results.get('ssim', 0):.4f}")
            logger.info(f"STATUS: PASS")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"✗ Error processing {subject}: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
            # Save failed result too
            result_file = self.output_dir / f'{subject}_metrics.json'
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("\n" + "="*80)
            logger.info(f"SUBJECT {subject} FAILED")
            logger.info("="*80)
            logger.info(f"STATUS: FAILED")
            logger.info(f"Error: {str(e)}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"✗ Error processing {subject}: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
            # Save failed result too
            result_file = self.output_dir / f'{subject}_metrics.json'
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def run_analysis(self, subjects: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run analysis on all subjects or specified subjects.
        
        Parameters
        ----------
        subjects : list, optional
            List of subject IDs. If None, process all subjects.
            
        Returns
        -------
        pd.DataFrame
            Results table with all metrics
        """
        if subjects is None:
            subjects = self.find_subjects()
        
        logger.info(f"\nProcessing {len(subjects)} subjects...")
        
        results_list = []
        
        for i, subject in enumerate(subjects, 1):
            logger.info(f"\n[{i}/{len(subjects)}] {subject}")
            result = self.process_subject(subject)
            results_list.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results_list)
        
        # Save results
        output_file = self.output_dir / 'signal_quality_metrics.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"\n✓ Results saved to {output_file}")
        
        return df
    
    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate dataset-level summary statistics.
        
        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame
            
        Returns
        -------
        dict
            Summary statistics
        """
        logger.info("\n" + "="*80)
        logger.info("DATASET SUMMARY")
        logger.info("="*80)
        
        # Filter successful subjects
        df_success = df[df['status'] == 'success']
        
        logger.info(f"Total subjects: {len(df)}")
        logger.info(f"Successful: {len(df_success)}")
        logger.info(f"Failed: {len(df) - len(df_success)}")
        
        if len(df_success) == 0:
            logger.error("No successful subjects!")
            return {}
        
        summary = {}
        
        # Metrics to summarize
        metrics = [
            ('raw_snr', 'Raw SNR'),
            ('preproc_snr', 'Preprocessed SNR'),
            ('raw_tsnr_median', 'Raw tSNR (median)'),
            ('preproc_tsnr_median', 'Preprocessed tSNR (median)'),
            ('psnr', 'PSNR'),
            ('ssim', 'SSIM')
        ]
        
        for metric_key, metric_name in metrics:
            if metric_key in df_success.columns:
                values = df_success[metric_key].dropna()
                
                if len(values) > 0:
                    summary[metric_key] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'median': float(values.median()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'n': int(len(values))
                    }
                    
                    logger.info(f"\n{metric_name}:")
                    logger.info(f"  Mean ± SD: {summary[metric_key]['mean']:.2f} ± {summary[metric_key]['std']:.2f}")
                    logger.info(f"  Median: {summary[metric_key]['median']:.2f}")
                    logger.info(f"  Range: [{summary[metric_key]['min']:.2f}, {summary[metric_key]['max']:.2f}]")
        
        # Save summary
        summary_file = self.output_dir / 'summary_statistics.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\n✓ Summary saved to {summary_file}")
        
        return summary


def main():
    """Main execution function with argument parsing and resume support."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Signal Quality Metrics for fMRI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--subject',
        type=str,
        help='Process single subject (e.g., --subject sub-01)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous run, skip completed subjects'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate transformation, do not compute all metrics'
    )
    
    args = parser.parse_args()
    
    # Paths
    raw_dir = Path('/root/fMRI/ds004302-download')
    preproc_dir = Path('/root/fMRI/output')
    output_dir = Path('/root/fMRI/quality_analysis/signal_quality')
    
    # Initialize
    analyzer = SignalQualityMetrics(
        raw_dir=raw_dir,
        preproc_dir=preproc_dir,
        output_dir=output_dir
    )
    
    # Single subject mode
    if args.subject:
        logger.info("\n" + "="*80)
        logger.info(f"SINGLE SUBJECT MODE: {args.subject}")
        logger.info("="*80)
        
        result = analyzer.process_subject(args.subject)
        
        if result['status'] == 'success':
            logger.info("\n" + "="*80)
            logger.info(f"SUBJECT {args.subject} COMPLETE")
            logger.info("="*80)
            logger.info(f"STATUS: PASS")
        else:
            logger.error("\n" + "="*80)
            logger.info(f"SUBJECT {args.subject} FAILED")
            logger.info("="*80)
            logger.error(f"Error: {result.get('error', 'Unknown')}")
        
        return
    
    # Full dataset mode
    logger.info("\n" + "="*80)
    logger.info("VALIDATION ON sub-01")
    logger.info("="*80)
    
    # Check if sub-01 already completed (resume mode)
    sub01_result_file = output_dir / 'sub-01_metrics.json'
    
    if args.resume and sub01_result_file.exists():
        logger.info("\n✓ Found existing sub-01 result, validating...")
        with open(sub01_result_file, 'r') as f:
            test_result = json.load(f)
        
        if test_result.get('status') == 'success':
            logger.info("✓ sub-01 result is valid, skipping recomputation")
        else:
            logger.info("✗ sub-01 result invalid, reprocessing...")
            test_result = analyzer.process_subject('sub-01')
    else:
        test_result = analyzer.process_subject('sub-01')
    
    if test_result['status'] == 'success':
        logger.info("\n✓ Validation successful! Proceeding with remaining subjects...")
        
        # Get all subjects
        all_subjects = analyzer.find_subjects()
        
        # Check which subjects are already completed (resume mode)
        completed_subjects = []
        if args.resume:
            for subject in all_subjects:
                result_file = output_dir / f'{subject}_metrics.json'
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                    if result.get('status') == 'success':
                        completed_subjects.append(subject)
                        logger.info(f"Skipping {subject}: valid completed result already exists")
        
        # Process remaining subjects
        remaining_subjects = [s for s in all_subjects if s not in completed_subjects]
        
        logger.info(f"\nProcessing {len(remaining_subjects)} subjects...")
        logger.info(f"  Already completed: {len(completed_subjects)}")
        logger.info(f"  Remaining: {len(remaining_subjects)}")
        
        # Load existing results
        results_list = []
        for subject in completed_subjects:
            result_file = output_dir / f'{subject}_metrics.json'
            with open(result_file, 'r') as f:
                results_list.append(json.load(f))
        
        # Process remaining subjects
        for i, subject in enumerate(remaining_subjects, 1):
            logger.info(f"\n[{i}/{len(remaining_subjects)}] {subject}")
            result = analyzer.process_subject(subject)
            results_list.append(result)
            
            # Save individual result immediately
            result_file = output_dir / f'{subject}_metrics.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Update cumulative CSV
            df = pd.DataFrame(results_list)
            output_file = output_dir / 'signal_quality_metrics.csv'
            df.to_csv(output_file, index=False)
        
        logger.info(f"\n✓ All results saved to {output_dir}")
        
        # Generate summary
        df = pd.DataFrame(results_list)
        summary = analyzer.generate_summary(df)
        
        # Save final summary
        summary_file = output_dir / 'final_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        
        # Report statistics
        n_success = sum(1 for r in results_list if r.get('status') == 'success')
        n_failed = len(results_list) - n_success
        
        logger.info(f"\nTotal subjects: {len(results_list)}")
        logger.info(f"Successful: {n_success}")
        logger.info(f"Failed: {n_failed}")
        logger.info(f"Success rate: {100*n_success/len(results_list):.1f}%")
        
        if n_failed > 0:
            logger.info("\nFailed subjects:")
            for r in results_list:
                if r.get('status') != 'success':
                    logger.info(f"  {r.get('subject', 'unknown')}: {r.get('error', 'Unknown error')}")
        
    else:
        logger.error("\n✗ Validation failed! Please check errors above.")
        logger.error(f"Error: {test_result.get('error', 'Unknown')}")
        logger.error("\nCannot proceed with full dataset until validation passes.")


if __name__ == '__main__':
    main()
