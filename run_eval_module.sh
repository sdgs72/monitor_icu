#!/bin/sh

set -x

EXPERIMENT="exp_2_h"

DATA_DIR="./data"
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

python code/main.py \
  --phase=inference \
  --batch_size=128 \
  --eval_data_split="test" \
  --eval_dataset_size=0 \
  --eval_checkpoint="experiments/exp_2_h/checkpoint_epoch050.model" \
  --checkpoint_dir="./experiments" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_test.log 2>&1 &
