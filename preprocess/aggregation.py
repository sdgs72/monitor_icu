import collections
import csv
import numpy as np
import os
from tqdm import tqdm

from absl import logging
from scipy.sparse import csr_matrix

# Total HADM IDs: 22049
VOC_SIZE = 4006  # In total of 4006 Events
WINDOW_LENGTH = 6  # 6 hours per block
SPLIT_DIR = "../data"
RAW_DATA = "../raw_data/MIMIC_FULL_BATCH.csv"
RAW_ADM_INFO = "../raw_data/MIMIC_ADM_INFO.csv"
VOCABULARY_FILE = "events_vocabulary.csv"

if not os.path.exists(VOCABULARY_FILE):
  item_voc = set([])
  with open(RAW_DATA, "r") as fp:
    reader = csv.DictReader(fp)
    for row in tqdm(reader, total=119982892 - 1):
      item_voc.add(row["EventType"] + row["ITEMID2"])

  with open(VOCABULARY_FILE, "w") as fp:
    writer = csv.writer(fp)
    for x in item_voc:
      writer.writerow([x])

events_vocabulary = {}
events_dict = {}

with open(VOCABULARY_FILE, "r") as fp:
  reader = csv.reader(fp)
  for i, x in enumerate(reader):
    events_vocabulary[x[0].strip()] = i
    events_dict[i] = x[0].strip()

# MIMIC_ADM_INFO.csv header
# ,HADM_ID,EventType,ITEMID2,TIME

hadm_length = {}
with open(RAW_ADM_INFO, "r") as fp:
  reader = csv.DictReader(fp)
  for x in reader:
    hadm_length[x["HADM_ID"]] = int(
        np.ceil((int(x["adm_length"]) + 1) / WINDOW_LENGTH))


def get_data_splits(dir_path):
  data = {}
  for x in ["train", "val", "test"]:
    data[x] = set([])
    full_path = os.path.join(dir_path, "%s_hadmids.csv" % x)
    with open(full_path, "r") as fp:
      reader = csv.DictReader(fp)
      for row in reader:
        data[x].add(row["HADM_ID"])
  return data


data = get_data_splits(SPLIT_DIR)

output = collections.defaultdict(dict)

with open(RAW_DATA, "r") as fp:
  reader = csv.DictReader(fp)
  hadm_id = None
  record = None
  for row in tqdm(reader, total=119982892 - 1):
    if row["HADM_ID"] != hadm_id:
      if hadm_id:
        if hadm_id in data["train"]:
          target = output["train"]
        elif hadm_id in data["val"]:
          target = output["val"]
        elif hadm_id in data["test"]:
          target = output["test"]
        else:
          raise ValueError("Unknown data split.")
        target[hadm_id] = csr_matrix(record)
      hadm_id = row["HADM_ID"]
      record = np.zeros((hadm_length[hadm_id], VOC_SIZE))

    key = row["EventType"] + row["ITEMID2"]
    block = int(row["TIME"]) // WINDOW_LENGTH
    if key not in events_vocabulary:
      continue
    try:
      record[block, events_vocabulary[key]] += 1
    except:
      # print("Events after discharge/death: %s" % row)
      pass

  if hadm_id in data["train"]:
    target = output["train"]
  elif hadm_id in data["val"]:
    target = output["val"]
  elif hadm_id in data["test"]:
    target = output["test"]
  else:
    raise ValueError("Unknown data split.")
  target[hadm_id] = csr_matrix(record)

print("Saving train.")
np.save("train_interval6_data", output["train"])

print("Saving test.")
np.save("test_interval6_data", output["test"])

print("Saving val.")
np.save("val_interval6_data", output["val"])
