from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import sklearn
from sklearn.externals import joblib


def sigmoid(x):
  return np.exp(-np.logaddexp(0, -x))


def update_metrics(y_true, y_score, phase, summary_writer=None, step=0):
  y_true = np.concatenate(y_true, axis=None)
  y_score = np.concatenate(y_score, axis=None)
  y_pred = y_score > 0.5

  logging.info("=" * 50)
  total = y_true.shape[0]
  corrects = np.sum(y_true == y_pred)

  accuracy = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
  if summary_writer is not None:
    summary_writer.add_scalar("%s/accuracy" % phase, accuracy, step)

  f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred)
  if summary_writer is not None:
    summary_writer.add_scalar("%s/f1" % phase, f1, step)

  roc_auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score)
  if summary_writer is not None:
    summary_writer.add_scalar("%s/roc_auc" % phase, roc_auc, step)

  ap = sklearn.metrics.average_precision_score(y_true=y_true, y_score=y_score)
  if summary_writer is not None:
    summary_writer.add_scalar("%s/average_precision" % phase, ap, step)

  pr_curve = sklearn.metrics.precision_recall_curve(
      y_true=y_true, probas_pred=sigmoid(y_score))
  if summary_writer is not None:
    summary_writer.add_pr_curve("%s/pr_curve" % phase, y_true, sigmoid(y_score),
                                step)

  logging.info("Total: %d", total)
  logging.info("Correct: %d", corrects)
  logging.info("Accuracy: %.4f", accuracy)
  logging.info("F1 Score: %.4f", f1)
  logging.info("ROC AUC Score: %.4f", roc_auc)
  logging.info("Average Precision: %.4f", ap)

  logging.info(
      "Classification Report:\n%s",
      sklearn.metrics.classification_report(
          y_true=y_true,
          y_pred=y_pred,
          target_names=["negative", "positive"],
          digits=4))
  logging.info("=" * 50)

  return accuracy, f1, roc_auc, ap


class Prediction:

  def __init__(self, block_size, history_window, prediction_window):
    self.block_size = block_size
    self.history_window = history_window
    self.prediction_window = prediction_window
    self.hadm_ids = []
    self.start_blocks = []
    self.end_blocks = []
    self.labels = []
    self.logits = []
    self.attentions = []
    return

  def add_prediction(self, data_info, logits, labels, attentions=None):
    hadm_ids, start_blocks, end_blocks = data_info

    self.hadm_ids.append(hadm_ids.numpy())
    self.start_blocks.append(start_blocks.numpy())
    self.end_blocks.append(end_blocks.numpy())

    self.logits.append(logits.numpy())
    self.labels.append(labels.numpy())

    if attentions is not None:
      self.attentions.append(attentions.numpy().squeeze(axis=-1))

    return

  def write_to_csv(self, filename):
    header = [
        "hadm_id", "block_size", "start_block", "end_block", "history_window",
        "prediction_window", "ground_truth", "logit", "pred_score", "prediction"
    ]

    hadm_ids = np.concatenate(self.hadm_ids)
    block_size = np.ones(hadm_ids.shape) * self.block_size
    start_blocks = np.concatenate(self.start_blocks)
    end_blocks = np.concatenate(self.end_blocks)
    history_window = np.ones(hadm_ids.shape) * self.history_window
    prediction_window = np.ones(hadm_ids.shape) * self.prediction_window
    labels = np.concatenate(self.labels)
    logits = np.concatenate(self.logits)
    pred_scores = sigmoid(logits)
    predictions = (logits > 0).astype(int)

    num_instances = hadm_ids.shape[0]
    assert num_instances == start_blocks.shape[0]
    assert num_instances == end_blocks.shape[0]
    assert num_instances == labels.shape[0]
    assert num_instances == logits.shape[0]

    data = [
        hadm_ids, block_size, start_blocks, end_blocks, history_window,
        prediction_window, labels, logits, pred_scores, predictions
    ]

    data = np.stack(data, axis=1)
    save_format = ["%d"] * 7 + ["%.6f", "%.6f", "%d"]

    if len(self.attentions) > 0:
      header.extend(
          ["attention_score@%d" % i for i in range(self.history_window)])
      attentions = np.concatenate(self.attentions, axis=0)

      logging.info("data dim: %s", data.shape)
      logging.info("attention dim: %s", attentions.shape)

      data = np.concatenate([data, attentions], axis=1)
      save_format.extend(["%.6f"] * self.history_window)
      assert num_instances == attentions.shape[0]

    delimiter = ","
    np.savetxt(
        filename,
        data,
        fmt=save_format,
        delimiter=delimiter,
        header=delimiter.join(header))

    return
