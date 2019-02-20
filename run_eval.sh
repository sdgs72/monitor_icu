#!/bin/sh

set -x

EXPERIMENT="death_trial"

DATA_DIR="./data"
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

python code/main.py \
  --phase=inference \
  --batch_size=128 \
<<<<<<< HEAD
  --eval_data_split=val \
=======
  --eval_data_split="val" \
>>>>>>> 86f8e4178c649c0fa64369be52d8f9ed6ca9a9a3
  --eval_dataset_size=0 \
  --checkpoint_dir="./experiments" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_val.log 2>&1 &

python code/main.py \
  --phase=inference \
  --batch_size=128 \
<<<<<<< HEAD
  --eval_data_split=test \
=======
  --eval_data_split="test" \
>>>>>>> 86f8e4178c649c0fa64369be52d8f9ed6ca9a9a3
  --eval_dataset_size=0 \
  --checkpoint_dir="./experiments" \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_test.log 2>&1 &
