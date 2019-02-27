#!/bin/sh

set -x

EXPERIMENT="debug1"

DATA_DIR="./data"
LOG_DIR="./debug"

mkdir -p ${LOG_DIR}

python code/main.py \
  --phase=inference \
  --batch_size=128 \
  --eval_data_split="test" \
  --eval_dataset_size=0 \
  --eval_checkpoint="debug/debug1/checkpoint_best_val_acc.model" \
  --checkpoint_dir="./debug" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_test.log 2>&1 &
