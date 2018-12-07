#!/bin/sh

set -x

EXPERIMENT="death_attentional_gru"

DATA_DIR="/afs/cs.pitt.edu/usr0/miz44/mimic_project/data/"
LOG_DIR="/afs/cs.pitt.edu/usr0/miz44/mimic_project/logs"

mkdir -p ${LOG_DIR}

python code/main.py \
  --phase=inference \
  --batch_size=128 \
  --data_split=val \
  --dataset_size=0 \
  --checkpoint_dir=/afs/cs.pitt.edu/usr0/miz44/mimic_project/experiments \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_val.log 2>&1 &

python code/main.py \
  --phase=inference \
  --batch_size=128 \
  --data_split=test \
  --dataset_size=0 \
  --checkpoint_dir=/afs/cs.pitt.edu/usr0/miz44/mimic_project/experiments \
  --experiment_name="${EXPERIMENT}" >> ${LOG_DIR}/${EXPERIMENT}_test.log 2>&1 &
