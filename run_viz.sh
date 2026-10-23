#!/usr/bin/env bash
set -euo pipefail

# Determine repository root relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

mkdir -p "$SCRIPT_DIR/results/figures"

echo "=========================================================================="
echo "Generating Publication Figures from Cross-Validation Metrics"
echo "Directory: $SCRIPT_DIR"
echo "=========================================================================="

python3 visualize_results.py "$@"
