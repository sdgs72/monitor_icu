import csv
import os


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


data_split = get_data_splits("../data/")

# HADM_ID,TotalBlocks,Discharge,Death,Sepsis
# 100001,25.0,25.0,-1.0,-1.0

with open("../data/labelTable.csv", "r") as fp, open("split_label_table.csv",
                                                     "w") as w_fp:
  reader = csv.reader(fp)
  header = next(reader)
  writer = csv.writer(w_fp)
  writer.writerow(header + ["DataSplit"])
  for row in reader:
    if row[0] in data_split["train"]:
      writer.writerow([int(float(x)) for x in row] + ["train"])
    elif row[0] in data_split["val"]:
      writer.writerow([int(float(x)) for x in row] + ["val"])
    elif row[0] in data_split["test"]:
      writer.writerow([int(float(x)) for x in row] + ["test"])
    else:
      raise ValueError("Invalid HADM_ID. %s" % row["HADM_ID"])
