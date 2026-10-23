"""
Sliding-Window Dataset module for Pure Mamba-BiLSTM pipeline (No GAT).
Extracts windowed BOLD timeseries tensors (200, 100) and windowed functional connectivity (200, 200).
"""
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import logging

logger = logging.getLogger("mamba_bilstm_pipeline")


def load_participants_data(
    tsv_path: Path,
    exclude_high_motion: bool = False,
    high_motion_subs: List[str] = None
) -> List[Dict]:
    """
    Load participant labels and metadata.
    Class 0: HC (Healthy Control)
    Class 1: SCZ (AVH- and AVH+)
    """
    if high_motion_subs is None:
        high_motion_subs = []

    records = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sub_id = row["participant_id"]
            group = row["group"]

            if exclude_high_motion and sub_id in high_motion_subs:
                continue

            if group == "HC":
                binary_label = 0
            elif group in ("AVH-", "AVH+"):
                binary_label = 1
            else:
                continue

            records.append({
                "participant_id": sub_id,
                "group": group,
                "binary_label": binary_label,
                "is_high_motion": sub_id in high_motion_subs
            })

    return records


class BOLDWindowDataset(Dataset):
    """
    PyTorch Dataset storing sliding BOLD window tensors and functional connectivity profiles.
    """
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "x": s["x"],        # (200, 100)
            "fc": s["fc"],      # (200, 200)
            "y": s["y"],        # scalar
            "subject": s["subject"],
            "window_idx": s["window_idx"]
        }


def generate_sliding_window_samples(
    roi_timeseries: np.ndarray,
    label: int,
    subject_id: str,
    window_size: int = 100,
    window_stride: int = 25
) -> List[Dict]:
    """
    Slices timeseries into sliding window tensors (200, 100) and computes
    windowed functional connectivity profiles (200, 200) without GAT.
    """
    n_rois, n_timepoints = roi_timeseries.shape
    samples = []
    w_idx = 0

    for start in range(0, n_timepoints - window_size + 1, window_stride):
        end = start + window_size
        window_raw = roi_timeseries[:, start:end]  # (200, 100)

        # Standardize each ROI across time in this window
        means = np.mean(window_raw, axis=1, keepdims=True)
        stds = np.std(window_raw, axis=1, keepdims=True) + 1e-8
        norm_window = (window_raw - means) / stds

        # Compute Pearson correlation matrix: (200, 200)
        corr = np.dot(norm_window, norm_window.T) / (norm_window.shape[1] - 1)
        np.fill_diagonal(corr, 0.0)

        # Fisher-z transform
        corr_clipped = np.clip(corr, -0.9999, 0.9999)
        fc_z = np.arctanh(corr_clipped)

        samples.append({
            "x": torch.tensor(norm_window, dtype=torch.float),
            "fc": torch.tensor(fc_z, dtype=torch.float),
            "y": label,
            "subject": subject_id,
            "window_idx": w_idx
        })
        w_idx += 1

    return samples


def get_balanced_sampler(samples: List[Dict]) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that balances HC and SCZ windows in every batch.
    """
    labels = [s["y"] for s in samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)

    sample_weights = [class_weights[y] for y in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
