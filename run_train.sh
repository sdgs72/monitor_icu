#!/bin/sh

set -x

TARGET="sepsis"
EXPERIMENT="${TARGET}_lstm"

DATA_DIR="/afs/cs.pitt.edu/usr0/miz44/mimic_project/data/"
LOG_DIR="/afs/cs.pitt.edu/usr0/miz44/mimic_project/logs"

mkdir -p ${LOG_DIR}

python code/main.py \
  --phase="train" \
  --model_type="rnn" \
  --rnn_type="lstm" \
  --nobidirectional \
  --batch_size=128 \
  --input_size=256 \
  --output_size=64 \
  --num_epochs=100 \
  --learning_rate=1e-2 \
  --data_split="train" \
  --data_dir="${DATA_DIR}" \
  --target_label="${TARGET}" \
  --history_window=8 \
  --prediction_window=2 \
  --dataset_size=0 \
  --rnn_layers=1 \
  --rnn_dropout=0 \
  --standardize \
  --checkpoint_dir="/afs/cs.pitt.edu/usr0/miz44/mimic_project/experiments" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_train.log 2>&1 &


