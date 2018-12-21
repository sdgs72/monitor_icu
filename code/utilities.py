from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import sklearn
from sklearn.externals import joblib


def sigmoid(x):
  return np.exp(-np.logaddexp(0, -x))


def update_metrics(metrics_dict, name, y_true, y_score, phase, summary_writer,
                   step):
  y_true = np.concatenate(y_true, axis=None)
  y_score = np.concatenate(y_score, axis=None)
  y_pred = y_score > 0.5

  logging.info("=" * 50)
  total = y_true.shape[0]
  corrects = np.sum((y_true == y_pred))

  accuracy = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
  summary_writer.add_scalar("%s/accuracy" % phase, accuracy, step)

  f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred)
  summary_writer.add_scalar("%s/f1" % phase, f1, step)

  roc_auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score)
  summary_writer.add_scalar("%s/roc_auc" % phase, roc_auc, step)

  ap = sklearn.metrics.average_precision_score(y_true=y_true, y_score=y_score)
  summary_writer.add_scalar("%s/average_precision" % phase, ap, step)

  pr_curve = sklearn.metrics.precision_recall_curve(
      y_true=y_true, probas_pred=sigmoid(y_score))

  metrics_dict[name] = {
      "total": total,
      "corrects": corrects,
      "accuracy": accuracy,
      "f1": f1,
      "roc_auc": roc_auc,
      "ap": ap,
      "pr_curve": pr_curve,
  }

  logging.info("Name: %s", name)
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

  return
