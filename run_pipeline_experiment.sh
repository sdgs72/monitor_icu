#!/bin/sh

set -x

TARGET="death"

EXPERIMENT="exp_name"

DATA_DIR="./data"
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

rm -rf ./experiments/exp_name
rm -rf ./logs/*

# See section 5.3..
# block size of 6 hours
# length of history window to 48 hours or 8 blocks 
# prediction window to 12 hours or 2 blocks
# initial learning rate set to 0.001. 
# We use a single hidden layer of size 32 in both LSTM and bidirectional LSTM experiments

python3 code/main.py \
  --phase="pipeline" \
  --model_type="rnn" \
  --rnn_type="lstm" \
  --rnn_bidirectional \
  --nouse_attention \
  --batch_size=128 \
  --input_size=256 \
  --rnn_hidden_size=256 \
  --num_epochs=25 \
  --learning_rate=1e-3 \
  --train_data_split="train" \
  --eval_data_split="val" \
  --data_dir="${DATA_DIR}" \
  --target_label="${TARGET}" \
  --block_size=6 \
  --history_window=8 \
  --prediction_window=4 \
  --train_dataset_size=0 \
  --eval_dataset_size=0 \
  --rnn_layers=1 \
  --rnn_dropout=0 \
  --standardize \
  --save_per_epochs=10 \
  --upper_bound_factor=5 \
  --fix_eval_dataset_seed=3750 \
  --checkpoint_dir="./experiments" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_train.log 2>&1
