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
#from sklearn.externals import joblib
import joblib

class MimicDataset(torch.utils.data.Dataset):

  def __init__(
      self,
      data_split,
      data_dir,
      block_size,
      target_label,
      history_window,
      prediction_window,
      dataset_size,
      pca_dim,
      pca_decomposer_path,
      standardize,
      standard_scaler_path,
      phase,
      upper_bound_factor,
      fix_dataset_seed=None,
  ):
    """Initializes dataset.

    Args:
      data_split: `train`, `val` or `test`.
      data_dir: Path to the data npy files and label csv file.
      block_size: Number of hours to be considered as one block.
      target_label: `death`, `discharge` or `sepsis`.
      history_window: Length of history events in blocks.
      prediction_window: Length of target event occurrence in future blocks.
      dataset_size: Size of generated dataset. If 0, then use all positive data
          with equal size of negative samples.
      pca_dim: Whether to perform pca for reduced dimensionality. If None then
          just return the original dimensionality.
      pca_decomposer_path: Path to precomputed (on `train`) pca decomposer.
      standardize: Whether to standardize inputs for zero mean and unit variance.
      standard_scaler_path: Path to precomputed (on `train`) standardizer.
      phase: `training` or `inference`.
      upper_bound_factor: number
    """
    self.data_name = "%s_interval1_data.npy"
    self.label_name = "hadm_infos.csv"

    self.data_dir = data_dir
    self.data_split = data_split
    self.block_size = block_size
    self.target_label = target_label
    self.history_window = history_window
    self.prediction_window = prediction_window
    self.dataset_size = dataset_size
    self.pca_dim = pca_dim
    self.pca_decomposer_path = pca_decomposer_path
    self.standardize = standardize
    self.standard_scaler_path = standard_scaler_path
    self.phase = phase
    # variable to set uppder bound
    self.upper_bound_factor = upper_bound_factor
    self.upper_bound_each_hadm_id = int(
        self.prediction_window * self.upper_bound_factor) + 1
    # variable recording the dropped hadmins
    #self.dropped_hadm_id = []
    self.fix_dataset_seed = fix_dataset_seed

    if self.fix_dataset_seed is None:
      self.random_generator = random
    else:
      self.random_generator = random.Random(self.fix_dataset_seed)

    if self.data_split not in ["train", "val", "test"]:
      raise ValueError(
          "Invalid `data_split`: Only `train`, `val` or `test` is supported.")

    dxh_file_path = os.path.join(self.data_dir, self.data_name % self.data_split)
    # DXH raw_data is in dictionary form
    self.raw_data = np.load(dxh_file_path, allow_pickle=True).item()
    logging.info("init: DXH file loaded for raw_data: %s", dxh_file_path)

    # print(f"DXH RAW_DATA[0] {self.raw_data}")

    self.labels, self.durations = self._load_labels()

    self.data = self._aggregate_raw_data()
    self.negatives, self.positives = self._generate_candidates()

    self.sample_list = self.resample()

    if self.pca_dim:
      self.decomposer, self.standard_scaler = self._preprocessing()
    return

  def __len__(self):
    return len(self.sample_list)

  def __getitem__(self, idx):
    hadm_id, (start_time, end_time), label = self.sample_list[idx]
    data = self.data[hadm_id][start_time:end_time, :]

    if data.shape[0] != self.history_window:
      raise AssertionError("Inconsistent data length.")

    if self.pca_dim:
      data = self.decomposer.transform(data)

      if self.standardize:
        data = self.standard_scaler.transform(data)

    return (data.astype(np.float32), label, (int(hadm_id), start_time,
                                             end_time))

  def _aggregate_raw_data(self):

    def _convert(mat):
      """Convert sparse matrix to np.ndarray, pad with zeros and reshape."""
      mat = mat.toarray()
      num_hours, voc_size = mat.shape
      if num_hours % self.block_size != 0:
        zeros = np.zeros((self.block_size - num_hours % self.block_size,
                          voc_size))
        padded = np.vstack((mat, zeros))
      else:
        padded = mat

      if padded.shape[0] % self.block_size:
        raise AssertionError("Invalid shape in padded data matrix.")

      return padded.reshape(-1, self.block_size, voc_size)

    logging.info("Number of admissions: %d", len(self.raw_data))
    logging.info("Preprocess dataset with block size of %d", self.block_size)
    aggregated_data = {}
    for hadm_id, mat in self.raw_data.items():
      aggregated_data[hadm_id] = np.sum(_convert(mat), axis=1)
      # logging.info("hadm_id: %s, shape: %s", hadm_id,
      #              aggregated_data[hadm_id].shape)
    logging.info("Preprocess completed.")

    return aggregated_data

  def resample(self):
    sample_list = self._sample_data()
    print(f"DXH _sample_data output list {len(sample_list)}")
    print(f"DXH _sample_data output list {sample_list}")

    logging.info("Resample dataset completed!")
    logging.info("First 10 records in the %s dataset:", self.data_split)
    for i in range(10):
      record = sample_list[i]
      logging.info("[%d] HADM_ID: %s, From %d to %d, Label: %s", i, record[0],
                   record[1][0], record[1][1], record[2])

    return sample_list

  def _preprocessing(self):
    decomposer, standard_scaler = None, None

    # aggregated_data = np.concatenate(list(self.data.values()), axis=0)
    aggregated_data = np.stack([np.sum(x, axis=0) for x in self.data.values()],
                               axis=0)
    logging.info("Feature dimension before PCA: %d", aggregated_data.shape[1])
    logging.info("Target dimension after PCA: %d", self.pca_dim)

    if self.phase == "training":
      decomposer = PCA(n_components=self.pca_dim)
      transformed_data = decomposer.fit_transform(aggregated_data)
      joblib.dump(decomposer, self.pca_decomposer_path)

      if self.standardize:
        standard_scaler = StandardScaler()
        standard_scaler.fit(transformed_data)
        joblib.dump(standard_scaler, self.standard_scaler_path)

    elif self.phase == "inference":
      if (not self.pca_decomposer_path or
          not os.path.exists(self.pca_decomposer_path)):
        raise AssertionError(
            "`pca_decomposer_path` is not defined or not found.")

      decomposer = joblib.load(self.pca_decomposer_path)

      if self.standardize:
        if (not self.standard_scaler_path or
            not os.path.exists(self.standard_scaler_path)):
          raise AssertionError(
              "`standard_scaler_path` is not defined or not found.")
        standard_scaler = joblib.load(self.standard_scaler_path)

    else:
      raise ValueError("Only `training` and `inference` are supported `phase`.")

    return decomposer, standard_scaler

  def _generate_candidates(self):
    full_negatives, full_positives = [], []

    logging.info("_generate_candidates[211]: Number of instances in labels: %d",
                 len(self.labels[self.target_label]))
    for hadm_id, label_time in self.labels[self.target_label].items():
      # (mingdaz): bug fix. If event happens after death/discharge, discard.
      logging.info(f"_generate_candidates: big looping -- ID:{hadm_id} Time:{label_time}")
      if label_time > self.durations[hadm_id]:
        continue

      negatives, positives = [], []

      if label_time < 0:  # all negatives

        #Mar 6: drop the ones with few effective events (these been regarded as abnormal ones)
        #cur_data = self.data[hadm_id]
        #if sum([1 for each in np.sum(cur_data,axis=1) if each == 0]) > cur_data.shape[0]*0.9:
        #  self.dropped_hadm_id.append(hadm_id)
        #  continue

        end_time = self.durations[hadm_id]
      else:
        end_time = label_time

      # DXH notice that start_time and end_time are in blocks here..
      for start_time in range(0, end_time - self.history_window + 1):
        logging.info(f"_generate_candidates: start history_window $$ ID:{hadm_id} StartTime:{start_time}")
        history_window = (start_time, start_time + self.history_window)
        prediction_window = range(
            start_time + self.history_window,
            start_time + self.history_window + self.prediction_window)
        label = label_time in prediction_window
        [negatives, positives][label].append((hadm_id, history_window, label))
      # Mar 6: pre-sample the negatives set of this hadm_id to be a set with a
      # size no larger than uppder_bound_each_hadm_id
      if len(negatives) > self.upper_bound_each_hadm_id:
        negatives = self.random_generator.sample(negatives,
                                                 self.upper_bound_each_hadm_id)

      full_negatives += negatives
      full_positives += positives

      if len(set(full_negatives)) != len(full_negatives):
        logging.warning("_generate_candidates: Duplicate samples in negative dataset.")

      if len(set(full_positives)) != len(full_positives):
        logging.warning("_generate_candidates: Duplicate samples in positive dataset.")

    logging.info(f"_generate_candidates: full_positives: {len(set(full_positives))} full_negatives: {len(set(full_negatives))}")
    return full_negatives, full_positives

  def _sample_data(self):
    print(f"******** DXH SAMPLING DATA START *********")
    if self.dataset_size == -1:
      logging.info("Using all possible candidates without sampling.")
      logging.warn(
          "WARN: The dataset might be extremely unbalanced and should be used in inference only."
      )
      print(f"******** DXH SAMPLING DATA DONE BLOCK_1 *********")
      return self.negatives + self.positives

    elif self.dataset_size == 0:
      self.dataset_size = len(self.positives) * 2
      logging.info("Dataset use default size: %d", len(self.positives) * 2)
      logging.info("  Use all %d positive samples", len(self.positives))
      logging.info("  Randomly choose %d negative samples from %d candidates",
                   len(self.positives), len(self.negatives))
      print(f"******** DXH SAMPLING DATA DONE BLOCK_2 *********")
      return self.random_generator.sample(
          self.negatives, k=len(self.positives)) + self.positives

    else:
      sample_list = self.random_generator.choices(
          self.negatives, k=self.dataset_size // 2)
      sample_list += self.random_generator.choices(
          self.positives, k=self.dataset_size - self.dataset_size // 2)          
      print(f"******** DXH SAMPLING DATA DONE BLOCK_3 *********")
      return sample_list

  def _load_labels(self):

    def _hour_to_block(str_hour):
      return int(str_hour) // self.block_size

    labels = {"death": {}, "discharge": {}, "sepsis": {}}
    durations = {}

    # csv header:
    # hadmId,admissionDuration,dischargeTime,deathTime,sepsisTime,dataSplit

    logging.info(f"_load_labels: DXH file loaded label_name {self.label_name}")
    with open(os.path.join(self.data_dir, self.label_name)) as fp:
      reader = csv.DictReader(fp)
      for row in reader:
        hadm_id = row["hadmId"]
        if row["dataSplit"] != self.data_split:
          continue
        durations[hadm_id] = _hour_to_block(row["admissionDuration"])
        labels["death"][hadm_id] = _hour_to_block(row["deathTime"])
        labels["discharge"][hadm_id] = _hour_to_block(row["dischargeTime"])
        labels["sepsis"][hadm_id] = _hour_to_block(row["sepsisTime"])

    return labels, durations
