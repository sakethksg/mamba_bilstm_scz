#!/usr/bin/env bash
export PATH=/usr/bin:/usr/local/bin:$PATH
export PYTHONPATH="/root/fMRI/mamba_bilstm_scz:$PYTHONPATH"
cd /root/fMRI/mamba_bilstm_scz
/usr/bin/python3 visualize_results.py
