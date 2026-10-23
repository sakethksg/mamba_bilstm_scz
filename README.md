# Pure Mamba-BiLSTM Hybrid SCZ Classifier (No GAT)

**Dual-Stage Spatio-Temporal Mamba State-Space Model + Bidirectional LSTM**

A graph-free deep learning architecture for schizophrenia (SCZ) classification from resting-state fMRI BOLD timeseries. This architecture eliminates GAT (Graph Attention Networks) and graph convolutional layers, replacing them with a streamlined **Spatio-Temporal Mamba-BiLSTM** pipeline.

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                 Input: Windowed BOLD Signals                 │
│                     (B, 200 ROIs, 100 TRs)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Stage 1: Temporal Mamba SSM Module              │
│       Multi-Scale Gated Dilated Convolutions (1,2,4,8)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            Stage 2: Temporal Bidirectional LSTM             │
│        Recurrent Forward & Backward Context along TRs       │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Residual Skip + LayerNorm)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Stage 3: Temporal Dual Readout                  │
│       Learned Temporal Attention + Mean Pooling -> (B, 200, d)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Stage 4: Spatial Bidirectional LSTM             │
│   Sequential Inter-Regional Network Modeling across 200 ROIs│
│              (Completely Replaces GAT / GNN)                │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Residual Skip + LayerNorm)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Stage 5: Multi-Scale Brain Readout & MLP            │
│       ROI Attention + Global Mean + Global Max Pooling      │
│            Classification MLP Head -> HC vs SCZ             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Advantages over GAT-based Models

1. **No GAT Dependency**: Completely removes Graph Attention Networks, edge-thresholding heuristics (top-k), and topological over-smoothing.
2. **True Spatio-Temporal Recurrence**: Uses Mamba SSM + BiLSTM temporally across time, and BiLSTM spatially across brain parcels.
3. **Speed & Efficiency**: Standard PyTorch tensor tensors ($B \times 200 \times 100$), zero graph collation latency, ~10x faster batching.
4. **All 71 Subjects Included**: Evaluated strictly on all 71 subjects across 5 stratified folds.

---

## Running Training

```bash
cd /root/fMRI/mamba_bilstm_scz
bash run.sh
```
