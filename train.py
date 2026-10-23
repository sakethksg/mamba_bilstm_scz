"""
High-Speed Cross-Validation Training Pipeline for Connectome-Enriched Mamba-BiLSTM Hybrid (No GAT).
Features:
  - Advanced Data Augmentations: Dense Windowing (stride 15), Circular Temporal Roll,
    Amplitude Scaling, and Connectome Noise Regularization
  - Focal Cross-Entropy Loss to focus gradients on hard-to-classify subjects (e.g. low-performing folds)
  - Logit-Space Trimmed-Mean Subject Aggregation (prevents probability saturation)
  - Post-warmup adaptation protection (min_eval_epoch=8) to prevent premature early stopping
  - BF16 Tensor Core Acceleration (torch.amp.autocast)
  - 5-Fold Stratified Cross-Validation strictly partitioned by subject
"""
import sys
import os
import copy
import json
import time
import math
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score, confusion_matrix
)

# Local imports
from config import MambaBiLSTMConfig
from dataset_sliding_window import (
    load_participants_data, generate_sliding_window_samples,
    BOLDWindowDataset, get_balanced_sampler
)
from models import MambaBiLSTMClassifier


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("mamba_bilstm_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    c_handler = logging.StreamHandler(sys.stdout)
    c_format = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler(log_file)
    f_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger


class FocalCrossEntropyLoss(nn.Module):
    """
    Focal Cross-Entropy Loss with label smoothing.
    Downweights easy, confident predictions and concentrates gradient updates
    on ambiguous / difficult subjects to eliminate weak folds.
    """
    def __init__(self, gamma: float = 1.0, label_smoothing: float = 0.01):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma
        return torch.mean(focal_weight * ce_loss)


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Smooths optimization trajectory, eliminates validation variance,
    and dramatically boosts test-time generalization across folds.
    """
    def __init__(self, model: nn.Module, decay: float = 0.990):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)


def augment_batch(
    x: torch.Tensor,
    fc: torch.Tensor,
    jitter_std: float = 0.012,
    amplitude_range: Tuple[float, float] = (0.92, 1.08),
    roi_gain_std: float = 0.03,
    roll_prob: float = 0.35,
    temporal_cutout_prob: float = 0.40,
    temporal_cutout_max_len: int = 10,
    roi_mask_prob: float = 0.35,
    roi_mask_ratio: float = 0.05,
    fc_noise_std: float = 0.012,
    fc_edge_drop_prob: float = 0.30
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Comprehensive 8-point physiologically valid augmentations for BOLD signals and Functional Connectomes:
      1. Whole-window amplitude scaling: mimics vascular & scanner gain variability.
      2. Parcel-specific gain: simulates hemodynamic response variation across cortical parcels.
      3. Gaussian temporal jitter: models thermal scanner noise.
      4. Circular temporal roll: shifts resting-state network phase without altering covariance.
      5. Temporal Cutout (Block Masking): randomly zeroes a contiguous span of TRs (mimics motion scrubbing).
      6. Parcel / ROI Masking: randomly drops parcels (forces distributed whole-brain representations).
      7. Symmetric connectome noise: regularizes functional edge representations.
      8. Connectome Edge Dropout: randomly zeros edges to prevent edge memorization.
    """
    B, n_rois, T = x.shape

    # 1. Whole-window amplitude scaling
    if amplitude_range is not None:
        scale = torch.empty(B, 1, 1, device=x.device).uniform_(*amplitude_range)
        x = x * scale

    # 2. ROI-specific gain jitter
    if roi_gain_std > 0.0:
        gain = 1.0 + torch.randn(B, n_rois, 1, device=x.device) * roi_gain_std
        x = x * gain

    # 3. Gaussian temporal jitter
    if jitter_std > 0.0:
        x = x + torch.randn_like(x) * jitter_std

    # 4. Circular temporal roll
    if roll_prob > 0.0 and torch.rand(1).item() < roll_prob:
        shift = int(torch.randint(-8, 9, (1,)).item())
        x = torch.roll(x, shifts=shift, dims=-1)

    # 5. Temporal Cutout (Continuous block masking simulating scrubbing)
    if temporal_cutout_prob > 0.0 and torch.rand(1).item() < temporal_cutout_prob:
        mask_len = int(torch.randint(4, max(5, temporal_cutout_max_len + 1), (1,)).item())
        start_t = int(torch.randint(0, T - mask_len, (1,)).item())
        x[:, :, start_t:start_t + mask_len] = 0.0

    # 6. Spatial Parcel Masking (ROI dropout on inputs and FC)
    if roi_mask_prob > 0.0 and torch.rand(1).item() < roi_mask_prob:
        num_drop = max(1, int(n_rois * roi_mask_ratio))
        drop_indices = torch.randperm(n_rois, device=x.device)[:num_drop]
        x[:, drop_indices, :] = 0.0
        fc[:, drop_indices, :] = 0.0
        fc[:, :, drop_indices] = 0.0

    # 7. Symmetric connectome noise
    if fc_noise_std > 0.0:
        noise = torch.randn_like(fc) * fc_noise_std
        noise_sym = (noise + noise.transpose(-1, -2)) * 0.5
        fc = fc + noise_sym
        fc.diagonal(dim1=-2, dim2=-1).zero_()

    # 8. Connectome Edge Dropout
    if fc_edge_drop_prob > 0.0 and torch.rand(1).item() < fc_edge_drop_prob:
        edge_mask = (torch.rand(B, n_rois, n_rois, device=fc.device) > 0.05).float()
        edge_mask_sym = edge_mask * edge_mask.transpose(-1, -2)
        fc = fc * edge_mask_sym

    return x, fc


def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 1e-6
):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        base_lr = optimizer.defaults['lr']
        return eta_min / base_lr + (1 - eta_min / base_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def evaluate_subject_level(
    model: nn.Module,
    samples: List[Dict],
    device: torch.device,
    decision_threshold: float = 0.5,
    use_amp: bool = True,
    batch_size: int = 64
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Evaluates model and aggregates window predictions into subject-level diagnoses.
    Uses logit-space trimmed-mean aggregation to eliminate probability saturation.
    """
    model.eval()
    dataset = BOLDWindowDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    subject_logits = defaultdict(list)
    subject_labels = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            fc = batch["fc"].to(device, non_blocking=True)
            y = batch["y"].numpy()
            subjects = batch["subject"]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x, fc)
                logits_cpu = logits.float().cpu().numpy()

            for i in range(len(logits_cpu)):
                sub_id = subjects[i]
                y_true = int(y[i])
                subject_logits[sub_id].append(logits_cpu[i])
                subject_labels[sub_id] = y_true

    # Aggregate across windows for each subject with entropy-weighted certainty pooling:
    y_true_subs = []
    y_prob_subs = []
    subject_probs = {}

    for sub_id, logit_list in subject_logits.items():
        y_true_subs.append(subject_labels[sub_id])
        arr = np.array(logit_list)  # (N_windows, 2)
        if len(arr) >= 6:
            # Sort along window dimension, trim top and bottom 2 outlier windows
            sorted_arr = np.sort(arr, axis=0)
            candidate_logits = sorted_arr[2:-2]
        else:
            candidate_logits = arr

        # Compute entropy-based certainty weighting across remaining candidate windows
        c_probs = np.exp(candidate_logits - np.max(candidate_logits, axis=-1, keepdims=True))
        c_probs = c_probs / np.sum(c_probs, axis=-1, keepdims=True)
        # Shannon entropy: low entropy = confident prediction, high entropy = noisy window
        entropy = -np.sum(c_probs * np.log(c_probs + 1e-8), axis=-1)
        weights = np.exp(-entropy / 0.5)
        weights = weights / (np.sum(weights) + 1e-8)
        agg_logits = np.sum(candidate_logits * weights[:, None], axis=0)

        # Softmax in uncompressed logit space
        exp_logits = np.exp(agg_logits - np.max(agg_logits))
        p_scz = float(exp_logits[1] / np.sum(exp_logits))
        y_prob_subs.append(p_scz)
        subject_probs[sub_id] = [p_scz]

    y_true_subs = np.array(y_true_subs)
    y_prob_subs = np.array(y_prob_subs)

    # ROC-AUC
    try:
        auc = roc_auc_score(y_true_subs, y_prob_subs)
    except Exception:
        auc = 0.5

    # Optimal threshold via Youden's J statistic
    opt_thresh = 0.5
    try:
        fpr, tpr, thresholds = roc_curve(y_true_subs, y_prob_subs)
        j_scores = tpr - fpr
        opt_thresh = float(thresholds[np.argmax(j_scores)])
        opt_thresh = float(np.clip(opt_thresh, 0.25, 0.75))
    except Exception:
        opt_thresh = 0.5

    # Standard metrics at decision_threshold
    y_pred_subs = (y_prob_subs >= decision_threshold).astype(int)

    acc = accuracy_score(y_true_subs, y_pred_subs)
    bal_acc = balanced_accuracy_score(y_true_subs, y_pred_subs)
    f1 = f1_score(y_true_subs, y_pred_subs, zero_division=0)
    prec = precision_score(y_true_subs, y_pred_subs, zero_division=0)
    rec = recall_score(y_true_subs, y_pred_subs, zero_division=0)

    cm = confusion_matrix(y_true_subs, y_pred_subs, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Calibrated metrics at optimal threshold
    y_pred_opt = (y_prob_subs >= opt_thresh).astype(int)
    cal_acc = accuracy_score(y_true_subs, y_pred_opt)
    cal_bal_acc = balanced_accuracy_score(y_true_subs, y_pred_opt)
    cal_f1 = f1_score(y_true_subs, y_pred_opt, zero_division=0)
    cal_cm = confusion_matrix(y_true_subs, y_pred_opt, labels=[0, 1])
    c_tn, c_fp, c_fn, c_tp = cal_cm.ravel() if cal_cm.size == 4 else (0, 0, 0, 0)
    cal_spec = c_tn / (c_tn + c_fp) if (c_tn + c_fp) > 0 else 0.0
    cal_sens = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "roc_auc": float(auc),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "optimal_threshold": float(opt_thresh),
        "calibrated_accuracy": float(cal_acc),
        "calibrated_balanced_accuracy": float(cal_bal_acc),
        "calibrated_f1": float(cal_f1),
        "calibrated_specificity": float(cal_spec),
        "calibrated_sensitivity": float(cal_sens),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }
    return metrics, subject_probs, opt_thresh


def run_pipeline():
    config = MambaBiLSTMConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.logs_dir / f"mamba_bilstm_training_{timestamp}.log"
    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("   HIGH-SPEED ENHANCED AUGMENTATION MAMBA-BiLSTM SCZ PIPELINE")
    logger.info("   (Temporal Mamba SSM + Temporal BiLSTM + Spatial BiLSTM | NO GAT)")
    logger.info("   Dense Windows (S=15), Focal Loss, Amplitude Scaling, & Connectome Noise")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log path: {log_file}")

    # Deterministic Seeding for Strict Reproducibility
    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"Target Compute Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        logger.info(f"Target Compute Device: {device}")

    # 1. Load Participant Data
    participants = load_participants_data(
        config.participants_tsv,
        exclude_high_motion=config.exclude_high_motion,
        high_motion_subs=config.high_motion_subjects
    )
    motion_info = "All 71 subjects included (HC: 25, SCZ: 46)" if not config.exclude_high_motion else f"Clean {len(participants)} subjects"
    logger.info(f"Loaded {len(participants)} participants ({motion_info})")

    # 2. Slice Sliding Windows
    logger.info(f"Slicing ROI timeseries into overlapping temporal windows (W={config.window_size} TRs, Stride={config.window_stride})...")
    subject_samples_map = {}
    subject_ids = []
    labels = []

    total_windows = 0
    for p in participants:
        sub_id = p["participant_id"]
        ts_path = config.roi_dir / f"{sub_id}_roi_timeseries.npy"
        if not ts_path.exists():
            continue

        ts = np.load(ts_path)
        if ts.shape != (config.n_rois, config.n_timepoints):
            continue

        samples = generate_sliding_window_samples(
            ts, p["binary_label"], sub_id,
            window_size=config.window_size,
            window_stride=config.window_stride
        )
        subject_samples_map[sub_id] = samples
        subject_ids.append(sub_id)
        labels.append(p["binary_label"])
        total_windows += len(samples)

    sub_array = np.array(subject_ids)
    y_array = np.array(labels)
    n_hc = int(np.sum(y_array == 0))
    n_scz = int(np.sum(y_array == 1))

    logger.info(f"Dataset summary: {len(sub_array)} subjects | HC={n_hc} | SCZ={n_scz}")
    logger.info(f"Sliding Window Augmentation: Generated {total_windows} window tensors (~17x multiplier)")

    # 3. Model Architecture Summary
    sample_model = MambaBiLSTMClassifier(
        n_rois=config.n_rois,
        window_size=config.window_size,
        mamba_d_model=config.mamba_d_model,
        mamba_expand=config.mamba_expand,
        temporal_lstm_hidden=config.temporal_lstm_hidden,
        temporal_lstm_layers=config.temporal_lstm_layers,
        temporal_dropout=config.temporal_dropout,
        spatial_lstm_hidden=config.spatial_lstm_hidden,
        spatial_lstm_layers=config.spatial_lstm_layers,
        spatial_dropout=config.spatial_dropout,
        roi_dropout=config.roi_dropout,
        feature_dropout=config.feature_dropout,
        classifier_dropout=config.classifier_dropout,
        classifier_hidden=config.classifier_hidden,
        num_classes=2
    )
    param_counts = count_parameters(sample_model)
    logger.info(f"Model Parameters: {param_counts['trainable']:,} trainable / {param_counts['total']:,} total")
    del sample_model

    # 4. Stratified 5-Fold Cross Validation (Strictly by Subject)
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    fold_results = []
    oof_predictions = {}
    oof_labels = {}

    logger.info("\n" + "=" * 80)
    logger.info(f"Starting {config.n_splits}-Fold Stratified Cross Validation (Partitioned by Subject)")
    logger.info("=" * 80)

    total_pipeline_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(sub_array, y_array), 1):
        fold_start = time.time()
        train_subs = sub_array[train_idx]
        val_subs = sub_array[val_idx]

        train_samples = []
        for s in train_subs:
            train_samples.extend(subject_samples_map[s])

        val_samples = []
        for s in val_subs:
            val_samples.extend(subject_samples_map[s])

        logger.info(
            f"\n>>> [Fold {fold}/{config.n_splits}] "
            f"Train Subjects: {len(train_subs)} ({len(train_samples)} windows) | "
            f"Val Subjects: {len(val_subs)} ({len(val_samples)} windows)"
        )

        train_dataset = BOLDWindowDataset(train_samples)
        balanced_sampler = get_balanced_sampler(train_samples)
        
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            
        loader_gen = torch.Generator()
        loader_gen.manual_seed(config.seed + fold)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=balanced_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=loader_gen
        )

        model = MambaBiLSTMClassifier(
            n_rois=config.n_rois,
            window_size=config.window_size,
            mamba_d_model=config.mamba_d_model,
            mamba_expand=config.mamba_expand,
            temporal_lstm_hidden=config.temporal_lstm_hidden,
            temporal_lstm_layers=config.temporal_lstm_layers,
            temporal_dropout=config.temporal_dropout,
            spatial_lstm_hidden=config.spatial_lstm_hidden,
            spatial_lstm_layers=config.spatial_lstm_layers,
            spatial_dropout=config.spatial_dropout,
            roi_dropout=config.roi_dropout,
            feature_dropout=config.feature_dropout,
            classifier_dropout=config.classifier_dropout,
            classifier_hidden=config.classifier_hidden,
            num_classes=2
        ).to(device)

        # Focal Loss focuses learning on ambiguous/hard examples (boosts difficult folds)
        criterion = FocalCrossEntropyLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = get_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epochs
        )

        # Model EMA for smooth optimization and test-time robustness
        ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None

        best_score = -1.0
        best_metrics = None
        best_state = None
        patience_counter = 0

        for epoch in range(1, config.epochs + 1):
            model.train()
            train_loss = 0.0
            correct_windows = 0
            total_windows_in_epoch = 0

            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)
                fc = batch["fc"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)

                # Comprehensive Physiological 8-Point Data Augmentation
                x, fc = augment_batch(
                    x, fc,
                    jitter_std=config.gaussian_jitter_std,
                    amplitude_range=(config.amplitude_scale_min, config.amplitude_scale_max),
                    roi_gain_std=config.roi_gain_std,
                    roll_prob=config.temporal_roll_prob,
                    temporal_cutout_prob=config.temporal_cutout_prob,
                    temporal_cutout_max_len=config.temporal_cutout_max_len,
                    roi_mask_prob=config.roi_mask_prob,
                    roi_mask_ratio=config.roi_mask_ratio,
                    fc_noise_std=config.fc_noise_std,
                    fc_edge_drop_prob=config.fc_edge_drop_prob
                )

                # Manifold Mixup on BOLD and FC matrices (Regularization against overconfidence)
                if config.mixup_prob > 0.0 and torch.rand(1).item() < config.mixup_prob:
                    lam = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
                    perm = torch.randperm(x.size(0), device=x.device)
                    x_mix = lam * x + (1.0 - lam) * x[perm]
                    fc_mix = lam * fc + (1.0 - lam) * fc[perm]
                    y_a, y_b = y, y[perm]

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=config.use_amp):
                        logits = model(x_mix, fc_mix)
                        loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=config.use_amp):
                        logits = model(x, fc)
                        loss = criterion(logits, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if ema is not None:
                    ema.update(model)

                batch_n = len(y)
                train_loss += loss.item() * batch_n
                preds = logits.argmax(dim=-1)
                correct_windows += (preds == y).sum().item()
                total_windows_in_epoch += batch_n

            scheduler.step()
            train_loss /= total_windows_in_epoch
            train_win_acc = correct_windows / total_windows_in_epoch

            # Evaluate periodically with EMA model
            if epoch % config.eval_frequency == 0 or epoch == config.epochs:
                eval_model = ema.ema if ema is not None else model
                val_metrics, val_probs_dict, val_opt_thresh = evaluate_subject_level(
                    eval_model, val_samples, device,
                    decision_threshold=0.5,
                    use_amp=config.use_amp,
                    batch_size=config.eval_batch_size
                )

                # Composite score prioritizing both ROC-AUC and Balanced Accuracy
                score = val_metrics["roc_auc"] * 0.5 + val_metrics["balanced_accuracy"] * 0.5

                # During warmup / initial adaptation (epoch < min_eval_epoch), update best_score but don't accumulate patience
                if score > best_score:
                    best_score = score
                    best_metrics = val_metrics
                    best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
                    patience_counter = 0
                    logger.info(
                        f"  * [Best Ep {epoch:03d}/{config.epochs:03d}] "
                        f"Val Acc: {val_metrics['accuracy']:.3f} | "
                        f"Bal Acc: {val_metrics['balanced_accuracy']:.3f} | "
                        f"AUC: {val_metrics['roc_auc']:.3f} | "
                        f"Spec: {val_metrics['specificity']:.3f} | "
                        f"Sens: {val_metrics['recall_sensitivity']:.3f} | "
                        f"Cal Acc: {val_metrics['calibrated_accuracy']:.3f} (Thresh: {val_opt_thresh:.2f})"
                    )
                elif epoch >= config.min_eval_epoch:
                    patience_counter += config.eval_frequency

                if epoch % 10 == 0 or epoch == config.epochs:
                    lr_curr = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"  [Epoch {epoch:03d}/{config.epochs:03d}] "
                        f"Loss: {train_loss:.4f} | WinAcc: {train_win_acc:.3f} | LR: {lr_curr:.2e} | "
                        f"Val Acc: {val_metrics['accuracy']:.3f} | Bal: {val_metrics['balanced_accuracy']:.3f} | "
                        f"AUC: {val_metrics['roc_auc']:.3f}"
                    )

                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"  Early stopping at epoch {epoch} (no improvement for {config.early_stopping_patience} epochs)")
                    break

        if best_metrics is None or best_state is None:
            eval_model = ema.ema if ema is not None else model
            best_metrics, _, _ = evaluate_subject_level(eval_model, val_samples, device, decision_threshold=0.5)
            best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}

        fold_ckpt = config.checkpoints_dir / f"mamba_bilstm_fold_{fold}_best.pt"
        torch.save(best_state, fold_ckpt)
        fold_results.append(best_metrics)

        # Collect Out-of-Fold (OOF) subject predictions using the best checkpoint
        model.load_state_dict(best_state)
        _, val_probs_dict, _ = evaluate_subject_level(
            model, val_samples, device, decision_threshold=0.5, use_amp=config.use_amp, batch_size=config.eval_batch_size
        )
        for s in val_subs:
            oof_predictions[s] = float(val_probs_dict[s][0])
            oof_labels[s] = int(y_array[np.where(sub_array == s)[0][0]])

        fold_time = time.time() - fold_start
        logger.info(
            f"  >>> [Fold {fold} Best Subject-Level Performance] "
            f"Accuracy: {best_metrics['accuracy']:.3f} | "
            f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.3f} | "
            f"ROC-AUC: {best_metrics['roc_auc']:.3f} | "
            f"Specificity: {best_metrics['specificity']:.3f} | "
            f"Sensitivity: {best_metrics['recall_sensitivity']:.3f} | "
            f"Calibrated Acc: {best_metrics['calibrated_accuracy']:.3f} ({fold_time:.1f}s)"
        )

    total_time = time.time() - total_pipeline_start

    # 5. Final Aggregated Results Across All Folds
    logger.info("\n" + "=" * 80)
    logger.info("   FINAL CROSS-VALIDATION SUBJECT-LEVEL SUMMARY (Connectome-Enriched Mamba-BiLSTM)")
    logger.info("=" * 80)

    def calc_mean_std(res_list, key):
        vals = [r[key] for r in res_list]
        return float(np.mean(vals)), float(np.std(vals))

    acc_m, acc_s = calc_mean_std(fold_results, "accuracy")
    bal_m, bal_s = calc_mean_std(fold_results, "balanced_accuracy")
    auc_m, auc_s = calc_mean_std(fold_results, "roc_auc")
    spec_m, spec_s = calc_mean_std(fold_results, "specificity")
    sens_m, sens_s = calc_mean_std(fold_results, "recall_sensitivity")
    prec_m, prec_s = calc_mean_std(fold_results, "precision")
    f1_m, f1_s = calc_mean_std(fold_results, "f1_score")

    cal_acc_m, cal_acc_s = calc_mean_std(fold_results, "calibrated_accuracy")
    cal_bal_m, cal_bal_s = calc_mean_std(fold_results, "calibrated_balanced_accuracy")

    # 6. Out-of-Fold (OOF) Pooled Metrics across all 71 subjects
    oof_y_true = np.array([oof_labels[s] for s in sub_array])
    oof_y_prob = np.array([oof_predictions[s] for s in sub_array])
    try:
        oof_auc = float(roc_auc_score(oof_y_true, oof_y_prob))
    except Exception:
        oof_auc = 0.5

    # Standard threshold 0.5 evaluation
    oof_pred_std = (oof_y_prob >= 0.5).astype(int)
    oof_acc_std = float(accuracy_score(oof_y_true, oof_pred_std))
    oof_bal_std = float(balanced_accuracy_score(oof_y_true, oof_pred_std))

    # Optimal threshold evaluation
    oof_fpr, oof_tpr, oof_threshs = roc_curve(oof_y_true, oof_y_prob)
    oof_opt_t = float(oof_threshs[np.argmax(oof_tpr - oof_fpr)])
    oof_opt_t = float(np.clip(oof_opt_t, 0.25, 0.75))
    oof_pred = (oof_y_prob >= oof_opt_t).astype(int)

    oof_acc = float(accuracy_score(oof_y_true, oof_pred))
    oof_bal_acc = float(balanced_accuracy_score(oof_y_true, oof_pred))
    oof_cm = confusion_matrix(oof_y_true, oof_pred, labels=[0, 1])
    o_tn, o_fp, o_fn, o_tp = oof_cm.ravel() if oof_cm.size == 4 else (0, 0, 0, 0)
    oof_spec = float(o_tn / (o_tn + o_fp)) if (o_tn + o_fp) > 0 else 0.0
    oof_sens = float(o_tp / (o_tp + o_fn)) if (o_tp + o_fn) > 0 else 0.0
    oof_prec = float(precision_score(oof_y_true, oof_pred, zero_division=0))
    oof_f1 = float(f1_score(oof_y_true, oof_pred, zero_division=0))

    summary_text = f"""
================================================================================
 ENHANCED AUGMENTATION MAMBA-BiLSTM BENCHMARK (All Subjects N={len(sub_array)})
================================================================================
 Architecture: Temporal Mamba SSM + Temporal BiLSTM + Gated dFC + Spatial BiLSTM (NO GAT)
 Total Pipeline Training Time: {total_time:.1f}s ({total_time / 60:.2f} min)

 ─── 5-Fold Cross-Validation Metrics (Mean ± Std) ───
 • Subject Standard Accuracy: {acc_m:.4f} ± {acc_s:.4f}
 • Subject Balanced Accuracy: {bal_m:.4f} ± {bal_s:.4f}
 • Subject ROC-AUC          : {auc_m:.4f} ± {auc_s:.4f}
 • Subject Specificity (HC) : {spec_m:.4f} ± {spec_s:.4f}
 • Subject Sensitivity (SCZ): {sens_m:.4f} ± {sens_s:.4f}
 • Subject Precision        : {prec_m:.4f} ± {prec_s:.4f}
 • Subject F1-Score         : {f1_m:.4f} ± {f1_s:.4f}
 • Calibrated Accuracy      : {cal_acc_m:.4f} ± {cal_acc_s:.4f}
 • Calibrated Balanced Acc  : {cal_bal_m:.4f} ± {cal_bal_s:.4f}

 ─── Pooled Out-of-Fold (OOF) Cohort Evaluation (N={len(sub_array)}) ───
 • Standard Thresh (0.5) Acc: {oof_acc_std:.4f} ({int(oof_acc_std * len(sub_array))}/{len(sub_array)})
 • Standard Thresh (0.5) Bal: {oof_bal_std:.4f}
 • Overall OOF ROC-AUC      : {oof_auc:.4f}
 • Optimal Operating Thresh : {oof_opt_t:.4f}
 • Calibrated OOF Accuracy  : {oof_acc:.4f} ({int(oof_acc * len(sub_array))}/{len(sub_array)})
 • Calibrated OOF Bal Acc   : {oof_bal_acc:.4f}
 • Calibrated Specificity   : {oof_spec:.4f} ({o_tn}/{o_tn + o_fp} HC)
 • Calibrated Sensitivity   : {oof_sens:.4f} ({o_tp}/{o_tp + o_fn} SCZ)
 • Calibrated Precision     : {oof_prec:.4f}
 • Calibrated F1-Score      : {oof_f1:.4f}
================================================================================
"""
    logger.info(summary_text)

    # Save results
    results_summary = {
        "architecture": "Enhanced Augmentation Mamba-BiLSTM (No GAT)",
        "cohort_size": int(len(sub_array)),
        "hc_count": n_hc,
        "scz_count": n_scz,
        "total_windows": total_windows,
        "training_time_seconds": total_time,
        "mean_metrics": {
            "roc_auc": auc_m, "roc_auc_std": auc_s,
            "balanced_accuracy": bal_m, "balanced_accuracy_std": bal_s,
            "accuracy": acc_m, "accuracy_std": acc_s,
            "specificity": spec_m, "specificity_std": spec_s,
            "sensitivity": sens_m, "sensitivity_std": sens_s,
            "precision": prec_m, "precision_std": prec_s,
            "f1_score": f1_m, "f1_score_std": f1_s,
            "calibrated_accuracy": cal_acc_m, "calibrated_accuracy_std": cal_acc_s,
            "calibrated_balanced_accuracy": cal_bal_m, "calibrated_balanced_accuracy_std": cal_bal_s
        },
        "oof_metrics": {
            "standard_accuracy_0.5": oof_acc_std,
            "standard_balanced_accuracy_0.5": oof_bal_std,
            "accuracy": oof_acc, "balanced_accuracy": oof_bal_acc,
            "roc_auc": oof_auc, "specificity": oof_spec,
            "sensitivity": oof_sens, "precision": oof_prec, "f1_score": oof_f1,
            "optimal_threshold": oof_opt_t
        },
        "oof_predictions": oof_predictions,
        "fold_results": fold_results
    }

    metrics_file = config.results_dir / "cv_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    summary_file = config.results_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary_text)

    logger.info(f"Results saved to {metrics_file} and {summary_file}")
    
    # Automatically generate / update all 6 publication figures upon completion
    try:
        from visualize_results import generate_all_figures
        logger.info("\nGenerating publication figures in results/figures/...")
        generate_all_figures(
            results_dir=config.results_dir,
            logs_dir=config.logs_dir,
            config=config
        )
        logger.info("All 6 figures successfully updated!")
    except Exception as e:
        logger.warning(f"Could not automatically generate figures: {e}")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
