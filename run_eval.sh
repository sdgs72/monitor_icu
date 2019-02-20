#!/bin/sh

set -x

EXPERIMENT="death_trial"

DATA_DIR="./data"
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

python code/main.py \
  --phase=inference \
  --batch_size=128 \
  --eval_data_split=val \
  --eval_dataset_size=0 \
  --checkpoint_dir="./experiments" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_val.log 2>&1 &

python code/main.py \
  --phase=inference \
  --batch_size=128 \
  --eval_data_split=test \
  --eval_dataset_size=0 \
  --checkpoint_dir="./experiments" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_test.log 2>&1 &
