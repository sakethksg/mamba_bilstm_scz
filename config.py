"""
Configuration for Fine-Tuned Mamba + Bidirectional LSTM (BiLSTM) SCZ Classification Pipeline.
No GAT / GNN graph layers. Connectome-Enriched Spatio-Temporal Mamba-BiLSTM.
Optimized for high-speed convergence, BF16 acceleration, and calibrated diagnostic accuracy.
Includes physiologically grounded data augmentations (dense temporal windowing, circular roll,
amplitude scaling, connectome regularization) and Focal Loss for low-performing fold recovery.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class MambaBiLSTMConfig:
    # --- Paths ---
    project_root: Path = Path("/root/fMRI")
    roi_dir: Path = Path("/root/fMRI/output/postprocessed/roi_timeseries")
    participants_tsv: Path = Path("/root/fMRI/ds004302-download/participants.tsv")
    output_dir: Path = Path("/root/fMRI/mamba_bilstm_scz")
    checkpoints_dir: Path = Path("/root/fMRI/mamba_bilstm_scz/checkpoints")
    results_dir: Path = Path("/root/fMRI/mamba_bilstm_scz/results")
    logs_dir: Path = Path("/root/fMRI/mamba_bilstm_scz/logs")

    # --- Cohort Selection ---
    # Include all 71 subjects (25 HC, 46 SCZ: 23 AVH-, 23 AVH+)
    high_motion_subjects: List[str] = field(default_factory=lambda: [
        "sub-02", "sub-22", "sub-27", "sub-28", "sub-34", "sub-45", "sub-69", "sub-77"
    ])
    exclude_high_motion: bool = False

    # --- Sliding Window Augmentation ---
    n_rois: int = 200
    n_timepoints: int = 341
    window_size: int = 100       # 100 TRs per temporal window
    window_stride: int = 12      # Stride of 12 TRs (21 windows/subject = 1,491 total windows, dense dynamic coverage)

    # --- Data Augmentation Parameters ---
    gaussian_jitter_std: float = 0.012     # BOLD temporal noise injection
    amplitude_scale_min: float = 0.92      # Vascular/scanner gain variation simulation
    amplitude_scale_max: float = 1.08
    roi_gain_std: float = 0.03             # Inter-parcel hemodynamic gain variation
    temporal_roll_prob: float = 0.35       # Circular time shift to break phase memorization
    temporal_cutout_prob: float = 0.40     # Transient block masking simulating motion scrubbing
    temporal_cutout_max_len: int = 10      # Up to 10 TRs masked
    roi_mask_prob: float = 0.35            # Random parcel dropout to enforce distributed representations
    roi_mask_ratio: float = 0.05           # 5% of parcels (10 ROIs)
    fc_noise_std: float = 0.012            # Symmetric connectome noise to prevent edge memorization
    fc_edge_drop_prob: float = 0.30        # Random connectome edge drop
    roi_dropout: float = 0.08              # Model-internal parcel feature dropout
    use_ema: bool = True                   # Exponential Moving Average of weights for test-time generalization
    ema_decay: float = 0.990               # Smooth weight averaging
    mixup_prob: float = 0.50               # Manifold Mixup probability on BOLD and FC
    mixup_alpha: float = 0.30              # Beta distribution parameter for Mixup interpolation

    # --- Temporal Mamba SSM Module ---
    mamba_d_model: int = 48       # Latent representation per parcel
    mamba_expand: int = 2         # Expansion factor in gated dilated SSM

    # --- Temporal BiLSTM Module ---
    temporal_lstm_hidden: int = 48 # Hidden units per direction (bidirectional = 96)
    temporal_lstm_layers: int = 1
    temporal_dropout: float = 0.20

    # --- Spatial BiLSTM Module (Replaces GAT) ---
    spatial_lstm_hidden: int = 48  # Hidden units per direction across brain regions (bidirectional = 96)
    spatial_lstm_layers: int = 1
    spatial_dropout: float = 0.25

    # --- Classifier Head & Regularization ---
    feature_dropout: float = 0.28
    classifier_dropout: float = 0.35
    classifier_hidden: int = 128
    label_smoothing: float = 0.08 # Prevents overconfident logit saturation on ambiguous/older subjects
    focal_gamma: float = 1.0      # Focal loss focusing parameter to prioritize hard-to-classify subjects

    # --- Fine-Tuning & Optimization ---
    n_splits: int = 5            # 5-fold cross-validation by SUBJECT
    batch_size: int = 16         # Robust batch size (uses only 3.0 GB VRAM)
    eval_batch_size: int = 32    # Vectorized evaluation
    epochs: int = 55             # Sufficient epochs for full convergence
    lr: float = 3.5e-4           # Optimal learning rate for AdamW
    weight_decay: float = 5.0e-4 # Enhanced L2 regularization preventing weight explosion
    warmup_epochs: int = 6       # Stable warmup before cosine decay
    min_eval_epoch: int = 6      # No early stopping during initial adaptation phase
    early_stopping_patience: int = 22  # Generous patience to allow hard folds to converge
    eval_frequency: int = 1      # Evaluate every epoch
    use_amp: bool = True         # BF16 mixed-precision acceleration
    num_workers: int = 2         # Multi-process data loading
    seed: int = 42
    device: str = "cuda:0"

    def __post_init__(self):
        for d in [self.output_dir, self.checkpoints_dir, self.results_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
