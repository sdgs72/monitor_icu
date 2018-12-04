from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import csv
import os
import numpy as np
import torch
import torch.utils.data
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


class MimicDataset(torch.utils.data.Dataset):

  def __init__(self,
               data_split,
               data_dir,
               target_label,
               history_window,
               prediction_window,
               dataset_size=10000,
               pca_dim=None,
               pca_decomposer_path=None,
               standardize=True,
               standard_scaler_path=None):
    """Initializes dataset.

    Args:
      data_split: `train`, `val` or `test`.
      data_dir: Path to the data npy files and label csv file.
      target_label: `death`, `discharge` or `sepsis`.
      history_window: Length of history events in blocks.
      prediction_window: Length of target event occurrence in future blocks.
      dataset_size: Size of generated dataset.
      pca_dim: Whether to perform pca for reduced dimensionality.
      pca_decomposer_path: Path to precomputed (on `train`) pca decomposer.
      standardize: Whether to standardize inputs for zero mean and unit variance.
      standard_scaler_path: Path to precomputed (on `train`) standardizer.
    """
    self.data_name = "%s_interval6_data.npy"
    self.label_name = "split_label_table.csv"

    self.data_dir = data_dir
    self.data_split = data_split
    self.target_label = target_label
    self.history_window = history_window  # Use 8 * 6 = 48h (2 days) history for model building
    self.prediction_window = prediction_window
    self.dataset_size = dataset_size
    self.pca_dim = pca_dim
    self.pca_decomposer_path = pca_decomposer_path
    self.standardize = standardize
    self.standard_scaler_path = standard_scaler_path

    if self.data_split not in ["train", "val", "test"]:
      raise ValueError(
          "Invalid `data_split`: Only `train`, `val` or `test` is supported.")

    self.data = np.load(
        os.path.join(self.data_dir, self.data_name % self.data_split)).item()

    self.labels, self.durations = self._load_labels()
    self.sample_list = self._sample_data()

    if self.pca_dim or self.standardize:
      self.decomposer, self.standard_scaler = self._preprocessing()
    return

  def __len__(self):
    return len(self.sample_list)

  def __getitem__(self, idx):
    hadm_id, (start_day, end_day), label = self.sample_list[idx]
    data = self.data[hadm_id][start_day:end_day, :].todense().A

    if self.pca_dim:
      data = self.decomposer.transform(data)

    if self.standardize:
      data = self.standard_scaler.transform(data)

    return data.astype(np.float32), label

  def _preprocessing(self):
    decomposer, standard_scaler = None, None

    aggregated_data = np.stack([np.sum(x, axis=0) for x in self.data.values()],
                               axis=0)

    if self.data_split == "train":
      decomposer = PCA(n_components=self.pca_dim)
      transformed_data = decomposer.fit_transform(aggregated_data)

      standard_scaler = StandardScaler()
      standard_scaler.fit(transformed_data)

      joblib.dump(decomposer, self.pca_decomposer_path)
      joblib.dump(standard_scaler, self.standard_scaler_path)
    else:
      assert self.pca_decomposer_path is not None, "Please specify `pca_decomposer_path`."
      assert self.standard_scaler_path is not None, "Please specify `standard_scaler_path`."

      decomposer = joblib.load(self.pca_decomposer_path)
      standard_scaler = joblib.load(self.standard_scaler_path)

    return decomposer, standard_scaler

  def _generate_candidates(self):
    full_negatives, full_positives = [], []
    for hadm_id, label_time in self.labels[self.target_label].items():
      negatives, positives = [], []
      if label_time < 0:  # all negatives
        end_time = self.durations[hadm_id]
      else:
        end_time = label_time

      for start_time in range(0, end_time - self.history_window + 1):
        history_window = (start_time, start_time + self.history_window)
        prediction_window = range(
            start_time + self.history_window,
            start_time + self.history_window + self.prediction_window)
        label = label_time in prediction_window
        [negatives, positives][label].append((hadm_id, history_window, label))
      full_negatives += negatives
      full_positives += positives

    return full_negatives, full_positives

  def _sample_data(self):
    negatives, positives = self._generate_candidates()

    sample_list = random.choices(negatives, k=self.dataset_size // 2)
    sample_list += random.choices(
        positives, k=self.dataset_size - self.dataset_size // 2)

    return sample_list

  def _load_labels(self):
    labels = {"death": {}, "discharge": {}, "sepsis": {}}
    durations = {}

    # csv header:
    # HADM_ID,TotalBlocks,Discharge,Death,Sepsis,DataSplit
    with open(os.path.join(self.data_dir, self.label_name)) as fp:
      reader = csv.DictReader(fp)
      for row in reader:
        hadmid = row["HADM_ID"]
        if row["DataSplit"] != self.data_split:
          continue
        duration = int(row["TotalBlocks"])
        durations[hadmid] = duration
        labels["death"][hadmid] = int(row["Death"])
        labels["discharge"][hadmid] = int(row["Discharge"])
        labels["sepsis"][hadmid] = int(row["Sepsis"])

    return labels, durations
