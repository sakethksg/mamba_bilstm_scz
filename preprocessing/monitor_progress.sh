#!/bin/bash
# Monitor signal quality metrics processing progress

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_FILE="$PROJECT_ROOT/logs/signal_quality_metrics_rerun.log"

echo "Monitoring signal quality metrics processing..."
echo "Log file: $LOG_FILE"
echo ""

while true; do
    # Count processed subjects
    SUCCESS=$(grep -c "Successfully processed" "$LOG_FILE" 2>/dev/null || echo "0")
    FAILED=$(grep -c "Error processing" "$LOG_FILE" 2>/dev/null || echo "0")
    TOTAL=$((SUCCESS + FAILED))
    
    # Get current subject being processed
    CURRENT=$(grep "Processing sub-" "$LOG_FILE" | tail -1 | grep -oP "sub-\d+" || echo "N/A")
    
    # Calculate progress
    if [ $TOTAL -gt 0 ]; then
        PERCENT=$((TOTAL * 100 / 71))
        echo "Progress: $TOTAL/71 subjects ($PERCENT%)"
        echo "  Successful: $SUCCESS"
        echo "  Failed: $FAILED"
        echo "  Current: $CURRENT"
        
        # Estimate time remaining (assuming 90 seconds per subject)
        REMAINING=$((71 - TOTAL))
        MINUTES=$((REMAINING * 90 / 60))
        echo "  Estimated time remaining: ~$MINUTES minutes"
    fi
    
    echo ""
    echo "Last 5 lines:"
    tail -5 "$LOG_FILE"
    echo ""
    echo "----------------------------------------"
    
    # Check if complete
    if grep -q "Analysis complete" "$LOG_FILE"; then
        echo "✓ Processing complete!"
        break
    fi
    
    sleep 60
done
