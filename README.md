# Connectome-Enriched Spatio-Temporal Mamba-BiLSTM for Schizophrenia Classification

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/Managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/OpenNeuro-ds004302-informational.svg)](https://openneuro.org/datasets/ds004302)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the **Connectome-Enriched Spatio-Temporal Mamba-BiLSTM** network for automated schizophrenia (SCZ) classification and neural biomarker discovery from resting-state functional Magnetic Resonance Imaging (rs-fMRI) blood-oxygen-level-dependent (BOLD) timeseries.

---

## Abstract & Overview

Deep learning models for neuroimaging connectomics frequently rely on Graph Attention Networks (GAT) or Graph Convolutional Networks (GCN) that depend on ad-hoc graph thresholding heuristics, introduce edge-selection bias, and are prone to topological over-smoothing across cortical parcels. 

This repository provides a graph-free, spatio-temporal deep learning framework that couples continuous-time **Mamba State-Space Models (SSM)** with **Bidirectional Long Short-Term Memory (BiLSTM)** networks:
- **Temporal Modeling**: Captures multi-scale hemodynamic dynamics ($1, 2, 4, 8$ TRs) and bidirectional temporal dependencies without attention matrix quadratic complexity.
- **Spatial Modeling**: Sequence-based whole-brain inter-regional representation learning across cortical parcels regularized by functional connectivity priors.
- **Physiologically Grounded Augmentation**: Mitigates scanner gain drift, subject motion artifacts, and phase memorization via dense temporal windowing, circular roll, amplitude scaling, and symmetric connectome noise.

---

## Key Contributions

- **Elimination of Graph Heuristics**: Eliminates top-$k$ functional connectivity thresholding and heuristic graph message passing, operating directly on standard tensor representations.
- **Dual-Stage Spatio-Temporal Modeling**: Integrates continuous-time linear state-space sequence modeling with bidirectional recurrent context over temporal dynamics and spatial parcel sequences.
- **Strict Subject-Level Cross-Validation**: Validated using 5-fold stratified cross-validation on all $N=71$ clinical participants with zero cross-fold window leakage.
- **Interpretable Connectome Biomarkers**: Includes automated biomarker extraction across Yeo 7 functional networks (Visual, Somatomotor, Dorsal Attention, Salience/Ventral Attention, Limbic, Control, Default Mode).

---

## Cohort & Dataset Specifications

Evaluated on the clinical rs-fMRI cohort from OpenNeuro ([ds004302](https://openneuro.org/datasets/ds004302)):

| Attribute | Specification |
|:---|:---|
| **Participants ($N=71$)** | 25 Healthy Controls (HC), 46 Patients with Schizophrenia (SCZ) |
| **Parcellation Atlas** | Schaefer 200 Cortical Parcels ($N_{\text{ROI}} = 200$) |
| **Functional Networks** | 7 Canonical Resting-State Networks (Yeo et al.) |
| **Temporal Length** | 341 TRs per subject ($\text{TR} = 2.0\,\text{s}$, $682\,\text{s}$ total scan duration) |
| **Windowing Strategy** | Sliding temporal window of 100 TRs ($200\,\text{s}$), stride = 12 TRs (21 windows/subject, 1,491 total dynamic windows) |

---

## Experimental Benchmark Results

Model performance evaluated via **5-Fold Stratified Cross-Validation** and **Pooled Out-of-Fold (OOF)** validation across all 71 participants:

| Metric | 5-Fold Cross-Validation (Mean ± SD) | Pooled Out-of-Fold (OOF) |
|:---|:---:|:---:|
| **Calibrated Balanced Accuracy** | **88.22% ± 11.08%** | **79.87%** |
| **Calibrated Accuracy** | **87.14% ± 10.50%** | **77.46% (55/71)** |
| **ROC-AUC** | **82.67% ± 17.41%** | **78.00%** |
| **Specificity (Healthy Control)** | **84.00% ± 14.97%** | **88.00% (22/25)** |
| **Sensitivity (Schizophrenia)** | **73.33% ± 37.58%** | **71.74% (33/46)** |
| **Precision** | **71.33% ± 36.35%** | **91.67%** |
| **F1-Score** | **72.28% ± 36.89%** | **80.49%** |

*All results are saved in [`results/cv_metrics.json`](results/cv_metrics.json) and [`results/summary.txt`](results/summary.txt).*

---

## Publication Figures

The evaluation suite automatically produces six publication-ready figures in both vector (`.pdf`) and raster (`.png`) formats:

| Figure | Description | Output Location |
|:---|:---|:---|
| **Figure 1** | Cross-Validation Performance Comparison across Folds | [`results/figures/fig1_cv_performance_comparison.pdf`](results/figures/fig1_cv_performance_comparison.pdf) |
| **Figure 2** | ROC Curves (Individual Folds + Mean ROC + OOF ROC) | [`results/figures/fig2_roc_curves.pdf`](results/figures/fig2_roc_curves.pdf) |
| **Figure 3** | Standard & Calibrated Confusion Matrices | [`results/figures/fig3_confusion_matrices.pdf`](results/figures/fig3_confusion_matrices.pdf) |
| **Figure 4** | Subject-Level Diagnostic Probability Distributions | [`results/figures/fig4_subject_diagnostic_probabilities.pdf`](results/figures/fig4_subject_diagnostic_probabilities.pdf) |
| **Figure 5** | Training Convergence & Loss/Accuracy Dynamics | [`results/figures/fig5_training_convergence.pdf`](results/figures/fig5_training_convergence.pdf) |
| **Figure 6** | Connectome Topological Biomarkers across 7 Functional Networks | [`results/figures/fig6_connectome_topological_biomarkers.pdf`](results/figures/fig6_connectome_topological_biomarkers.pdf) |

---

## Repository Structure

```
mamba_bilstm_scz/
├── config.py                 # Hyperparameter configuration dataclass
├── dataset_sliding_window.py # Sliding-window BOLD dataset loader and augmentations
├── mamba_bilstm_temporal.py  # Temporal Mamba SSM and BiLSTM modules
├── models.py                 # Full hybrid classifier architecture
├── train.py                  # 5-fold cross-validation training pipeline
├── visualize_results.py      # Publication figure generator and metrics evaluator
├── run.sh                    # Automated pipeline execution script
├── run_viz.sh                # Visualization reproduction script
├── pyproject.toml            # Project metadata and uv dependency configuration
├── requirements.txt          # Standard pip dependencies
├── checkpoints/              # Trained 5-fold model weights (*.pt)
├── results/
│   ├── cv_metrics.json       # Quantitative performance results
│   ├── summary.txt           # Summary metrics report
│   └── figures/              # Publication figures (.pdf / .png)
└── logs/
    └── run_latest.log        # Full benchmark training log
```

---

## Installation & Environment Setup

### Environment Setup with `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) provides rapid, deterministic environment management:

```bash
# Clone the repository
git clone https://github.com/sakethksg/mamba_bilstm_scz.git
cd mamba_bilstm_scz

# Create and activate a virtual environment
uv venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# Synchronize dependencies from lockfile
uv sync
```

### Alternative Setup with `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

## Reproducing Results

### 1. Training & Cross-Validation

To execute the 5-fold cross-validation pipeline:

```bash
# Using the shell script:
bash run.sh

# Or using uv:
uv run python train.py

# Or directly with Python:
python train.py
```

### 2. Generating Publication Visualizations

To reproduce all publication figures from cross-validation predictions:

```bash
bash run_viz.sh
# Or:
uv run python visualize_results.py
```

---

## Configuration Reference

Key experimental parameters can be adjusted in [`config.py`](config.py):

```python
# Cohort & Windowing
n_rois = 200                    # Schaefer 200 cortical parcels
window_size = 100               # 100 TR temporal window length (200s)
window_stride = 12              # Stride for dense dynamic windowing (21 windows/subject)

# Architectural Dimensions
mamba_d_model = 48              # Latent representation per parcel
mamba_expand = 2                # Dilated state expansion factor
temporal_lstm_hidden = 48       # Hidden units per direction (bidirectional = 96)
spatial_lstm_hidden = 64        # Spatial hidden units per direction (bidirectional = 128)

# Optimization & Regularization
learning_rate = 3e-4            # Cosine annealing learning rate schedule
weight_decay = 1e-4             # AdamW weight decay
use_ema = True                  # Exponential moving average of model weights (decay = 0.990)
focal_loss_gamma = 1.5          # Focal loss focusing parameter
```

---


## License & Data Acknowledgments

This project is released under the [MIT License](LICENSE). 

Data was obtained from the OpenNeuro database under accession number [ds004302](https://openneuro.org/datasets/ds004302).
