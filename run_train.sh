#!/bin/sh

set -x

EXPERIMENT="trial_3"

DATA_DIR="/afs/cs.pitt.edu/usr0/miz44/mimic_project/data/"
LOG_DIR="/afs/cs.pitt.edu/usr0/miz44/mimic_project/logs"

mkdir -p ${LOG_DIR}

python code/main.py \
  --phase="train" \
  --batch_size=512 \
  --input_size=256 \
  --output_size=256 \
  --bidirectional \
  --num_epochs=10 \
  --learning_rate=1e-2 \
  --data_split="train" \
  --data_dir="${DATA_DIR}" \
  --target_label="discharge" \
  --history_window=8 \
  --prediction_window=2 \
  --dataset_size=0 \
  --standardize \
  --checkpoint_dir="/afs/cs.pitt.edu/usr0/miz44/mimic_project/experiments" \
  --rnn_type="lstm" \
  --rnn_layers=1 \
  --rnn_dropout=0 \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_train.log 2>&1 &


