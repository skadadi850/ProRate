#!/usr/bin/env bash
# Retrains one ProRate arm from release/data/train. Output goes to outputs/train/<arm>/stage1.pt.
#   bash train.sh protrig_s8k_rw0p06_link3_s1
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
arm="${1:?arm name, one of the files in release/train_commands/}"
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TQDM_DISABLE=1
mkdir -p logs outputs/train
mapfile -t ARGS < "release/train_commands/$arm.txt"
python scripts/train_protrig.py "${ARGS[@]}" > "logs/train_$arm.log" 2>&1
echo "wrote outputs/train/$arm/stage1.pt"
