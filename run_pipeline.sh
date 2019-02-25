#!/bin/sh

set -x

TARGET="death"
EXPERIMENT="debug1"

DATA_DIR="./data"
LOG_DIR="./debug"

mkdir -p ${LOG_DIR}

python code/main.py \
  --phase="pipeline" \
  --model_type="lr" \
  --rnn_type="lstm" \
  --rnn_bidirectional \
  --nouse_attention \
  --batch_size=128 \
  --input_size=256 \
  --rnn_hidden_size=256 \
  --num_epochs=50 \
  --learning_rate=1e-3 \
  --train_data_split="train" \
  --eval_data_split="val" \
  --data_dir="${DATA_DIR}" \
  --target_label="${TARGET}" \
  --block_size=3 \
  --history_window=112 \
  --prediction_window=8 \
  --train_dataset_size=0 \
  --eval_dataset_size=0 \
  --rnn_layers=1 \
  --rnn_dropout=0 \
  --standardize \
  --save_per_epochs=10 \
  --checkpoint_dir="./debug" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_train.log 2>&1 &
