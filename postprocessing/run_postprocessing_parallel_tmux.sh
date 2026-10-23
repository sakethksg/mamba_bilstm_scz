#!/bin/bash
################################################################################
# Post-Processing Pipeline Runner - PARALLEL VERSION with TMUX
################################################################################
# 
# This script runs the parallel post-processing pipeline in a tmux session.
# Optimized for multi-core servers (96 cores, 1TB RAM).
#
# Usage:
#   ./run_postprocessing_parallel_tmux.sh
#
################################################################################

# Configuration
SESSION_NAME="postprocessing"
PROJECT_DIR="/root/fMRI"
SCRIPT_DIR="$PROJECT_DIR/scripts/python"
LOG_DIR="$PROJECT_DIR/logs/postprocessing"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/tmux_session_parallel_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "POST-PROCESSING PIPELINE - PARALLEL VERSION - TMUX SESSION LAUNCHER"
echo "================================================================================"
echo ""
echo "Session Name: $SESSION_NAME"
echo "Project Dir:  $PROJECT_DIR"
echo "Log File:     $LOG_FILE"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "ERROR: tmux is not installed!"
    echo "Please install tmux: apt-get install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "WARNING: Session '$SESSION_NAME' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session:    tmux attach -t $SESSION_NAME"
    echo "  2. Kill and restart:             tmux kill-session -t $SESSION_NAME"
    echo "  3. Create new session:           Change SESSION_NAME in this script"
    echo ""
    exit 1
fi

# Display server resources
echo "================================================================================"
echo "SERVER RESOURCES"
echo "================================================================================"
echo ""
echo "CPU Cores: $(nproc)"
echo "Memory:    $(free -h | grep Mem | awk '{print $2}')"
echo ""
echo "================================================================================"

# Create new tmux session
echo "Creating new tmux session..."
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Set working directory
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR" C-m

# Start logging
tmux send-keys -t "$SESSION_NAME" "exec > >(tee -a '$LOG_FILE') 2>&1" C-m

# Print header
tmux send-keys -t "$SESSION_NAME" "echo '================================================================================'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'POST-PROCESSING PIPELINE - PARALLEL VERSION'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '================================================================================'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Started: $(date)'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Session: $SESSION_NAME'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Log:     $LOG_FILE'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Parallel jobs: 20'" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m

# Run the parallel pipeline
tmux send-keys -t "$SESSION_NAME" "python3 $SCRIPT_DIR/postprocess_fmri_parallel.py" C-m

# Print completion message
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME" "echo '================================================================================'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'PIPELINE COMPLETE'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '================================================================================'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Finished: $(date)'" C-m

echo ""
echo "✓ TMUX session created successfully!"
echo ""
echo "================================================================================"
echo "HOW TO MANAGE THE TMUX SESSION"
echo "================================================================================"
echo ""
echo "1. ATTACH TO SESSION (view execution):"
echo "   $ tmux attach -t $SESSION_NAME"
echo ""
echo "2. DETACH FROM SESSION (keep running in background):"
echo "   Press: Ctrl+B then D"
echo ""
echo "3. LIST ALL TMUX SESSIONS:"
echo "   $ tmux ls"
echo ""
echo "4. KILL THE SESSION:"
echo "   $ tmux kill-session -t $SESSION_NAME"
echo ""
echo "5. VIEW LOG FILE:"
echo "   $ tail -f $LOG_FILE"
echo ""
echo "================================================================================"
echo "PARALLEL PROCESSING BENEFITS"
echo "================================================================================"
echo ""
echo "✓ Processing 20 subjects simultaneously"
echo "✓ Utilizing multiple CPU cores efficiently"
echo "✓ Estimated completion time: 30-60 minutes (vs 6-12 hours sequential)"
echo "✓ All output is being logged to: $LOG_FILE"
echo "✓ You can safely close this terminal"
echo ""
echo "================================================================================"
