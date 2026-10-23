# Pure Mamba-BiLSTM Hybrid SCZ Classifier (No GAT)

**Dual-Stage Spatio-Temporal Mamba State-Space Model + Bidirectional LSTM for Resting-State fMRI**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance, graph-free deep learning architecture for schizophrenia (SCZ) classification using resting-state fMRI BOLD timeseries. This architecture completely eliminates Graph Attention Networks (GAT), Graph Convolutional Networks (GCN), and top-$k$ edge thresholding heuristics, replacing them with an efficient **Spatio-Temporal Mamba-BiLSTM** pipeline.

---

## Architecture Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Input: Windowed BOLD Timeseries                      │
│                  (Batch Size, 200 ROIs, 100 TRs)                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Stage 1: Temporal Mamba SSM Module                       │
│    Multi-Scale Gated Dilated Convolutions (Dilations: 1, 2, 4, 8 TRs)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Stage 2: Temporal Bidirectional LSTM                     │
│    Recurrent Forward & Backward Context along TRs                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Residual Skip Connection + LayerNorm)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Stage 3: Temporal Dual Readout                           │
│    Learned Temporal Attention + Mean Pooling -> (B, 200 ROIs, d_model) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Stage 4: Spatial Bidirectional LSTM                      │
│    Sequential Inter-Regional Network Modeling across 200 Brain Parcels │
│                  (Completely Replaces GAT / GNN)                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Residual Skip Connection + LayerNorm)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Stage 5: Multi-Scale Brain Readout & MLP                 │
│    ROI Attention + Global Mean + Global Max Pooling                    │
│    Classification MLP Head -> Healthy Control (HC) vs Schizophrenia    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Key Highlights & Advantages

1. **Graph-Free Architecture**: Eliminates graph attention mechanisms, topological over-smoothing, and ad-hoc functional connectivity edge-thresholding heuristics.
2. **True Spatio-Temporal Modeling**: 
   - *Temporally*: Continuous-time State-Space dynamics (Mamba SSM) combined with forward/backward recurrent memory (BiLSTM) across TRs.
   - *Spatially*: Bidirectional recurrence across ordered anatomical parcels (Schaefer 200 atlas) with dynamic functional connectivity gating.
3. **High Efficiency & Speed**: Built on standard PyTorch 3D tensors (`B, 200, 100`) without graph collation bottlenecks, achieving ~10x faster batch processing and full BF16/FP16 mixed precision acceleration.
4. **Rigorous Cohort Evaluation**: Strict 5-fold stratified cross-validation evaluated across all 71 subjects (25 HC, 46 SCZ).

---

## Project Structure

```
mamba_bilstm_scz/
├── config.py                 # Central hyperparameter & pipeline configuration
├── dataset_sliding_window.py # Sliding-window BOLD dataset loader & augmentations
├── mamba_bilstm_temporal.py  # Temporal Mamba SSM + Temporal BiLSTM encoder blocks
├── models.py                 # Full hybrid classifier & spatial sequence architecture
├── train.py                  # 5-fold stratified CV training & validation engine
├── visualize_results.py      # Publication-quality plotting & biomarker analysis
├── run.sh                    # Automated training pipeline script
├── run_viz.sh                # Visualization generator script
├── pyproject.toml            # Project specification & uv configuration
├── requirements.txt          # Standard pip dependencies
├── results/
│   ├── cv_metrics.json       # Quantitative performance results (per-fold & OOF)
│   ├── summary.txt           # Text summary of model evaluation
│   └── figures/              # Publication figures (PDF & PNG format)
└── logs/                     # Training run logs and fold checkpoints
```

---

## Installation & Setup

### Option 1: Fast Setup with `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) is an extremely fast Python package manager.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/mamba_bilstm_scz.git
cd mamba_bilstm_scz

# 2. Create and activate a virtual environment
uv venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows

# 3. Synchronize dependencies
uv sync
# Alternatively:
# uv pip install -r requirements.txt
```

### Option 2: Standard `pip` and `venv`

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Training the Full 5-Fold Pipeline

Run the automated training script (handles GPU setup, logging, and metrics):

```bash
# Using the bash launcher:
bash run.sh

# Or directly with uv:
uv run python train.py

# Or within an activated environment:
python train.py
```

Optional training flags:
- Specify GPU: `CUDA_VISIBLE_DEVICES=0 bash run.sh`

### 2. Generating Publication Visualizations

Generate all 6 diagnostic figures (ROC curves, confusion matrices, fold performance, convergence trajectories, and connectome biomarkers):

```bash
# Using the bash launcher:
bash run_viz.sh

# Or with uv:
uv run python visualize_results.py
```

Generated plots will be saved to `results/figures/`:
- `fig1_cv_performance_comparison.{png,pdf}`
- `fig2_roc_curves.{png,pdf}`
- `fig3_confusion_matrices.{png,pdf}`
- `fig4_subject_diagnostic_probabilities.{png,pdf}`
- `fig5_training_convergence.{png,pdf}`
- `fig6_connectome_topological_biomarkers.{png,pdf}`

---

## Experimental Benchmark Results

Evaluated on $N=71$ subjects (25 Healthy Controls, 46 Schizophrenia) via 5-Fold Stratified Cross-Validation:

| Metric | 5-Fold CV (Mean ± Std) | Pooled Out-of-Fold (OOF) |
|---|:---:|:---:|
| **Calibrated Balanced Accuracy** | **88.22% ± 11.08%** | **79.87%** |
| **Calibrated Accuracy** | **87.14% ± 10.50%** | **77.46% (55/71)** |
| **ROC-AUC** | **82.67% ± 17.41%** | **78.00%** |
| **Specificity (HC)** | **84.00% ± 14.97%** | **88.00% (22/25)** |
| **Sensitivity (SCZ)** | **73.33% ± 37.58%** | **71.74% (33/46)** |
| **Precision** | **71.33% ± 36.35%** | **91.67%** |
| **F1-Score** | **72.28% ± 36.89%** | **80.49%** |

---

## Configuration & Hyperparameters

Key settings can be modified in [config.py](file:///c:/Research/mamba_bilstm_scz/config.py):
- `window_size`: 100 TRs (~200s window)
- `window_stride`: 12 TRs (21 windows per subject, 1,491 total dynamic windows)
- `mamba_d_model`: 48 latent channels
- `mamba_expand`: 2x expansion factor
- `temporal_lstm_hidden`: 48 units per direction (bidirectional = 96)
- `spatial_lstm_hidden`: 64 units per direction (bidirectional = 128)
- `learning_rate`: $3 \times 10^{-4}$ with Cosine Annealing and EMA decay (0.990)
- `augmentations`: Temporal roll, Gaussian jitter, ROI masking, BOLD amplitude scaling, and symmetric FC edge noise.

---

## Citation & License

This project is licensed under the MIT License.
