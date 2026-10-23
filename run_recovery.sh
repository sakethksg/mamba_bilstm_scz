#!/usr/bin/env bash
export PATH=/usr/bin:/usr/local/bin:$PATH
export PYTHONPATH="/root/fMRI/mamba_bilstm_scz:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/fMRI/mamba_bilstm_scz
/usr/bin/python3 /root/.gemini/antigravity-ide/brain/7e6b4b34-c899-443a-9f22-dd47d3f7888d/scratch/recover_folds_3_4.py 2>&1 | tee /root/fMRI/mamba_bilstm_scz/logs/recovery_folds_3_4.log
