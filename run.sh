#!/usr/bin/env bash
set -e

export PATH=/usr/bin:/usr/local/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$SCRIPT_DIR/logs/run_latest.log"
RUN_LOG="$SCRIPT_DIR/logs/run_${TIMESTAMP}.log"

echo "=========================================================================="
echo "Starting High-Speed Mamba-BiLSTM Hybrid SCZ Classification Pipeline"
echo "  Architecture: Connectome-Enriched Mamba SSM + BiLSTM (No GAT)"
echo "Timestamp: $(date)"
echo "Directory: $SCRIPT_DIR"
echo "Log file: $RUN_LOG"
echo "=========================================================================="

python3 train.py 2>&1 | tee -a "$RUN_LOG" | tee "$MAIN_LOG"
