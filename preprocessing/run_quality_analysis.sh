#!/bin/bash
# Setup and run script for fMRI Quality Metrics Framework

set -e

echo "=========================================="
echo "fMRI Quality Metrics Framework"
echo "Setup and Execution Script"
echo "=========================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RAW_DIR="$PROJECT_ROOT/ds004302-download"
PREPROC_DIR="$PROJECT_ROOT/output"
OUTPUT_DIR="$PROJECT_ROOT/quality_analysis"

# Check if directories exist
if [ ! -d "$RAW_DIR" ]; then
    echo "Error: Raw data directory not found: $RAW_DIR"
    exit 1
fi

if [ ! -d "$PREPROC_DIR" ]; then
    echo "Error: Preprocessed data directory not found: $PREPROC_DIR"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Raw data: $RAW_DIR"
echo "  Preprocessed data: $PREPROC_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Install dependencies
echo "=========================================="
echo "Installing Python dependencies..."
echo "=========================================="

pip install --upgrade pip
pip install -r "$PROJECT_ROOT/requirements.txt"

echo ""
echo "Dependencies installed successfully!"
echo ""

# Run quality metrics analysis
echo "=========================================="
echo "Running Quality Metrics Analysis..."
echo "=========================================="

python "$SCRIPT_DIR/../python/fmri_quality_metrics.py"

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output structure:"
    echo "  - Metrics: $OUTPUT_DIR/metrics/"
    echo "  - Statistics: $OUTPUT_DIR/statistics/"
    echo "  - Plots: $OUTPUT_DIR/plots/"
    echo ""
    
    # Run advanced visualizations if metrics file exists
    METRICS_FILE="$OUTPUT_DIR/metrics/quality_metrics.csv"
    if [ -f "$METRICS_FILE" ]; then
        echo "=========================================="
        echo "Generating Advanced Visualizations..."
        echo "=========================================="
        python "$SCRIPT_DIR/../python/advanced_visualizations.py"
        echo ""
        echo "Advanced visualizations complete!"
    fi
    
    echo ""
    echo "=========================================="
    echo "All analyses completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "Error: Analysis failed. Please check the error messages above."
    exit 1
fi
