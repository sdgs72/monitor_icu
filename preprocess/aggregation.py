import collections
import csv
import numpy as np
import os
from tqdm import tqdm

from absl import logging
from scipy.sparse import csr_matrix

# Total HADM IDs: 22049
VOC_SIZE = 4010  # Original 4006 Events + AKI 0~3 = 4010 In total
WINDOW_LENGTH = 1  # 1 hour per block
SPLIT_DIR = "../data"
RAW_DATA = SPLIT_DIR + "/raw_data/MIMIC_FULL_BATCH.csv"
VOCABULARY_FILE = "../data/events_vocabulary.csv"
HADM_INFO_FILE = "../data/hadm_infos.csv"


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


data_split = get_data_splits(SPLIT_DIR)

if not os.path.exists(VOCABULARY_FILE) or not os.path.exists(HADM_INFO_FILE):
  item_voc = set([])
  sepsis_time = {}
  death_time = {}
  discharge_time = {}

  all_hadm_ids = set([])

  with open(RAW_DATA, "r") as fp:
    reader = csv.DictReader(fp)
    for row in tqdm(reader):
      event = row["EventType"] + row["ITEMID2"]
      item_voc.add(event)

      hadm_id = row["HADM_ID"]
      all_hadm_ids.add(hadm_id)

      time = int(row["TIME"])
      if event == "Sepsis1":
        sepsis_time[hadm_id] = time
      elif event == "Death0":
        discharge_time[hadm_id] = time
      elif event == "Death1":
        death_time[hadm_id] = time

  item_voc -= set(["Sepsis1", "Death0", "Death1"])

  with open(VOCABULARY_FILE, "w") as fp:
    writer = csv.writer(fp)
    for x in item_voc:
      writer.writerow([x])

  hadm_length = {}

  with open(HADM_INFO_FILE, "w") as fp:
    writer = csv.writer(fp)
    writer.writerow([
        "hadmId",
        "admissionDuration",
        "dischargeTime",
        "deathTime",
        "sepsisTime",
        "dataSplit",
    ])
    for x in all_hadm_ids:
      if x in data_split["train"]:
        x_split = "train"
      elif x in data_split["val"]:
        x_split = "val"
      elif x in data_split["test"]:
        x_split = "test"
      else:
        raise ValueError("Unknown data split.")

      hadm_length[x] = max(0 if x not in discharge_time else discharge_time[x],
                           0 if x not in death_time else death_time[x]),

      if hadm_length[x] == 0:
        raise AssertionError("Zero length admission: %s" % x)

      writer.writerow([
          x,
          "%d" % hadm_length[x],
          "%d" % (discharge_time[x] if x in discharge_time else -1),
          "%d" % (death_time[x] if x in death_time else -1),
          "%d" % (sepsis_time[x] if x in sepsis_time else -1),
          x_split,
      ])

events_vocabulary = {}
events_dict = {}

with open(VOCABULARY_FILE, "r") as fp:
  reader = csv.reader(fp)
  for i, x in enumerate(reader):
    events_vocabulary[x[0].strip()] = i
    events_dict[i] = x[0].strip()

hadm_length = {}
with open(HADM_INFO_FILE, "r") as fp:
  reader = csv.DictReader(fp)
  for row in reader:
    hadm_length[row["hadmId"]] = int(
        np.ceil((int(row["admissionDuration"]) + 1) / WINDOW_LENGTH))

# MIMIC_ADM_INFO.csv header
# ,HADM_ID,EventType,ITEMID2,TIME

# hadm_length = {}
# with open(RAW_ADM_INFO, "r") as fp:
#   reader = csv.DictReader(fp)
#   for x in reader:
#     hadm_length[x["HADM_ID"]] = int(
#         np.ceil((int(x["adm_length"]) + 1) / WINDOW_LENGTH))

output = collections.defaultdict(dict)

with open(RAW_DATA, "r") as fp:
  reader = csv.DictReader(fp)
  hadm_id = None
  record = None
  for row in tqdm(reader):
    if row["HADM_ID"] != hadm_id:
      if hadm_id:
        if hadm_id in data_split["train"]:
          target = output["train"]
        elif hadm_id in data_split["val"]:
          target = output["val"]
        elif hadm_id in data_split["test"]:
          target = output["test"]
        else:
          raise ValueError("Unknown data split.")
        target[hadm_id] = csr_matrix(record)
      hadm_id = row["HADM_ID"]
      record = np.zeros((hadm_length[hadm_id], VOC_SIZE))

    key = row["EventType"] + row["ITEMID2"]
    block = int(row["TIME"]) // WINDOW_LENGTH
    if key not in events_vocabulary:
      # print("%s not in events vocabulary." % key)
      continue

    try:
      record[block, events_vocabulary[key]] += 1
    except:
      # print("Events after discharge/death: %s" % row)
      pass

  if hadm_id in data_split["train"]:
    target = output["train"]
  elif hadm_id in data_split["val"]:
    target = output["val"]
  elif hadm_id in data_split["test"]:
    target = output["test"]
  else:
    raise ValueError("Unknown data split.")
  target[hadm_id] = csr_matrix(record)

print("Saving train.")
np.save("../data/train_interval1_data", output["train"])

print("Saving test.")
np.save("../data/test_interval1_data", output["test"])

print("Saving val.")
np.save("../data/val_interval1_data", output["val"])
