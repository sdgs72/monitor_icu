#!/bin/sh

set -x

EXPERIMENT="balanced_selfattention"

DATA_DIR="./data"
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

python code/main.py \
  --phase=inference \
  --batch_size=128 \
  --eval_data_split="test" \
  --eval_dataset_size=-1 \
  --eval_checkpoint="experiments/balanced_selfattention/checkpoint_best_f1_on_val_epoch002.model" \ # Change this to your desired model...
  --checkpoint_dir="./experiments" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_test.log 2>&1 
