#!/usr/bin/env bash
set -euo pipefail
cd ~/fMRI

# Kill both background fMRIPrep jobs if this script is interrupted
trap 'echo "Interrupted — killing child jobs"; kill 0' SIGINT SIGTERM

export FREESURFER_HOME=/opt/freesurfer
export FS_FREESURFERENV_NO_OUTPUT=1

set +u
set +e
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
set -e
set -u

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=6
export NUMEXPR_NUM_THREADS=6

BIDS_DIR=~/fMRI/ds004302-download
OUT_DIR=~/fMRI/output
LICENSE=~/fMRI/license.txt

# --- Abort if NUMA topology isn't what we expect ---
if ! numactl --hardware | grep -q "available: 2 nodes"; then
  echo "ERROR: expected 2 NUMA nodes, aborting." >&2
  exit 1
fi

# --- Check current free memory per node before committing --mem-mb ---
echo "--- Current memory state (check before finalizing --mem-mb) ---"
free -h
numactl --hardware | grep free

# --- Build subject list (robust to filenames), alternate across nodes ---
mapfile -t ALL_SUBS < <(find "$BIDS_DIR" -maxdepth 1 -type d -name 'sub-*' -printf '%f\n' | sed 's/sub-//' | sort)
N=${#ALL_SUBS[@]}
if (( N == 0 )); then
  echo "ERROR: no subjects found under $BIDS_DIR — check BIDS_DIR path." >&2
  exit 1
fi

NODE0_SUBS=()
NODE1_SUBS=()
for i in "${!ALL_SUBS[@]}"; do
  if (( i % 2 == 0 )); then
    NODE0_SUBS+=("${ALL_SUBS[$i]}")
  else
    NODE1_SUBS+=("${ALL_SUBS[$i]}")
  fi
done
echo "Total subjects: $N  ->  Node0: ${#NODE0_SUBS[@]}  Node1: ${#NODE1_SUBS[@]}"

mkdir -p ~/fMRI/work/numa0 ~/fMRI/work/numa1 ~/fMRI/logs

# --- MEM_MB: set this based on the free-memory check above, not a guess ---
# NOTE: numactl --membind is unavailable in this container (blocked by seccomp:
# "set_mempolicy: Operation not permitted"). We rely on --cpunodebind only,
# which still pins each fMRIPrep process's compute to a socket; the kernel's
# first-touch allocator will place most of that process's memory on the same
# node under normal (non-memory-pressured) conditions. With ~738 GiB free and
# 300 GB requested per node (600 GB total), there is comfortable headroom.
MEM_MB=300000   # placeholder — confirm against current free per node above

( cd ~/fMRI/work/numa0 && \
numactl --cpunodebind=0 \
fmriprep \
  "$BIDS_DIR" "$OUT_DIR" participant \
  --participant-label "${NODE0_SUBS[@]}" \
  --fs-license-file "$LICENSE" \
  --work-dir ~/fMRI/work/numa0 \
  --fs-no-reconall \
  --nprocs 32 \
  --omp-nthreads 6 \
  --mem-mb "$MEM_MB" \
  --output-spaces MNI152NLin2009cAsym:res-2 anat \
  --random-seed 42 \
  --resource-monitor \
  --notrack \
  -v -v \
) 2>&1 | tee ~/fMRI/logs/fmriprep_numa0.log &
PID0=$!

( cd ~/fMRI/work/numa1 && \
numactl --cpunodebind=1 \
fmriprep \
  "$BIDS_DIR" "$OUT_DIR" participant \
  --participant-label "${NODE1_SUBS[@]}" \
  --fs-license-file "$LICENSE" \
  --work-dir ~/fMRI/work/numa1 \
  --fs-no-reconall \
  --nprocs 32 \
  --omp-nthreads 6 \
  --mem-mb "$MEM_MB" \
  --output-spaces MNI152NLin2009cAsym:res-2 anat \
  --random-seed 42 \
  --resource-monitor \
  --notrack \
  -v -v \
) 2>&1 | tee ~/fMRI/logs/fmriprep_numa1.log &
PID1=$!

wait $PID0 $PID1
echo "Both NUMA batches finished."