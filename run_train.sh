#!/bin/sh

set -x

TARGET="death"
EXPERIMENT="${TARGET}_2"

DATA_DIR="/afs/cs.pitt.edu/usr0/miz44/mimic_project/data/"
LOG_DIR="/afs/cs.pitt.edu/usr0/miz44/mimic_project/logs"

mkdir -p ${LOG_DIR}

python code/main.py \
  --verbosity=1 \
  --phase="train" \
  --model_type="attentional_lr" \
  --batch_size=8 \
  --input_size=128 \
  --output_size=128 \
  --bidirectional \
  --num_epochs=10 \
  --learning_rate=1e-2 \
  --data_split="train" \
  --data_dir="${DATA_DIR}" \
  --target_label="${TARGET}" \
  --history_window=8 \
  --prediction_window=2 \
  --dataset_size=0 \
  --standardize \
  --checkpoint_dir="/afs/cs.pitt.edu/usr0/miz44/mimic_project/experiments" \
  --rnn_type="lstm" \
  --rnn_layers=1 \
  --rnn_dropout=0 \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_train.log 2>&1 &


