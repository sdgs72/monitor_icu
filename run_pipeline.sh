#!/bin/sh

set -x

TARGET="death"
EXPERIMENT="exp_2_h"

DATA_DIR="./data"
LOG_DIR="./logs"
:
mkdir -p ${LOG_DIR}

python3 code/main.py \
  --phase="pipeline" \
  --model_type="rnn" \
  --rnn_type="lstm" \
  --rnn_bidirectional \
  --use_attention \
  --batch_size=128 \
  --input_size=256 \
  --rnn_hidden_size=128 \
  --num_epochs=50 \
  --learning_rate=1e-3 \
  --train_data_split="train" \
  --eval_data_split="val" \
  --data_dir="${DATA_DIR}" \
  --target_label="${TARGET}" \
  --block_size=6 \
  --history_window=56 \
  --prediction_window=4 \
  --train_dataset_size=0 \
  --eval_dataset_size=0 \
  --rnn_layers=1 \
  --rnn_dropout=0 \
  --standardize \
  --save_per_epochs=10 \
  --checkpoint_dir="./experiments" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_train.log 2>&1
