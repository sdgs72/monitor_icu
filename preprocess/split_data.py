import csv
import numpy as np
import os
import random

random.seed(3750)

csv_path = "../data/raw_data/MIMIC_ADM_INFO.csv"
root_path = "../data"

train, val, test = [], [], []


def write_to_csv(data_split, output_csv):
  with open(output_csv, "w") as fp:
    writer = csv.writer(fp)
    writer.writerow(["HADM_ID"])
    for hadm_id in data_split:
      writer.writerow([hadm_id[0]])
  return


def analyze(data_split):
  lengths = [int(x[1]) for x in data_split]
  return np.mean(lengths), np.std(lengths)


with open(csv_path, "r") as fp:
  reader = csv.DictReader(fp, delimiter=",")
  for record in reader:
    item = (record["HADM_ID"], record["adm_length"])
    random_val = random.random()
    if random_val < 0.6:
      train.append(item)
    elif random_val < 0.8:
      val.append(item)
    else:
      test.append(item)

print("Admission Lengths:")
print("train: %.2f +/- %.2f" % analyze(train))
print("val: %.2f +/- %.2f" % analyze(val))
print("test: %.2f +/- %.2f" % analyze(test))

write_to_csv(train, os.path.join(root_path, "train_hadmids.csv"))
write_to_csv(val, os.path.join(root_path, "val_hadmids.csv"))
write_to_csv(test, os.path.join(root_path, "test_hadmids.csv"))
