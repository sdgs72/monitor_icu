from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import json
import os
import numpy as np
import sys
import sklearn
import time
import torch

from model_builder import RNNModel
from dataset import MimicDataset

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "phase", "train", ["train", "inference"],
    "Specifies whether to train a model or run inference on pretrained models.")
flags.DEFINE_integer("batch_size", 512, "Number of instances in a batch.")
flags.DEFINE_integer("input_size", 256, "Number of features of the input.")
flags.DEFINE_integer(
    "output_size", 256,
    "Number of the features of the output embedding from the rnn module.")
flags.DEFINE_boolean("bidirectional", False,
                     "Whether to use bidirectional RNN model.")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs for training.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for model training.")
flags.DEFINE_enum("data_split", "train", ["train", "val", "test"],
                  "Specifies the `train`, `val` or `test` data being used.")
flags.DEFINE_string("data_dir", "../data/", "Directory path to data.")
flags.DEFINE_enum("target_label", "death", ["discharge", "death", "sepsis"], "")
flags.DEFINE_integer("history_window", 8,
                     "Number of blocks (6h/block) in history.")
flags.DEFINE_integer("prediction_window", 2,
                     "Number of blocks in future for prediction.")
flags.DEFINE_integer(
    "dataset_size", 0,
    "Number of instances in the each epoch. If 0 use auto mode.")
flags.DEFINE_boolean("standardize", True, "Whether to standardize input data.")
flags.DEFINE_string("checkpoint_dir", "./",
                    "Directory where trained models are stored.")
flags.DEFINE_string("experiment_name", None, "Identifies the experiments.")
flags.DEFINE_enum("rnn_type", "lstm", ["lstm", "gru"],
                  "Type of RNN modules for experiments.")
flags.DEFINE_integer("rnn_layers", 1, "Number of layers tacked in RNN modules.")
flags.DEFINE_float("rnn_dropout", 0.0, "Dropout rate in the RNN modules.")


def train(configs):
  root_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)

  with open(os.path.join(root_dir, "running_configs.cfg"), "w") as fp:
    fp.write("python ")
    for i, x in enumerate(sys.argv):
      logging.info("%d: %s", i, x)
      fp.write("%s \\\n" % x)

  train_dataset = MimicDataset(
      data_split=FLAGS.data_split,
      data_dir=configs["data_dir"],
      target_label=configs["target_label"],
      history_window=configs["history_window"],
      prediction_window=configs["prediction_window"],
      dataset_size=FLAGS.dataset_size,
      pca_dim=configs["input_size"],
      pca_decomposer_path=os.path.join(root_dir, "pca_decomposer.skmodel"),
      standardize=configs["standardize"],
      standard_scaler_path=os.path.join(root_dir, "standard_scaler.skmodel"),
  )

  train_loader = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=FLAGS.batch_size,
      shuffle=True,
      drop_last=False,
  )

  model = RNNModel(
      input_size=configs["input_size"],
      hidden_size=configs["output_size"],
      rnn_type=configs["rnn_type"],
      num_layers=configs["rnn_layers"],
      dropout=configs["rnn_dropout"],
      bidirectional=configs["bidirectional"],
  )

  logging.info("Training dataset size: %d", len(train_dataset))
  logging.info("Batches in each epoch: %d", len(train_loader))

  model.train()

  criterion = torch.nn.BCEWithLogitsLoss()

  logging.info("Model parameters: %s", model.parameters())
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

  logging.info("Initializatoin complete. Start training.")
  try:
    for epoch in range(FLAGS.num_epochs):
      start_time = time.time()
      for step, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
          logits, _last_embedding, _full_embedding = model(inputs)
          loss = criterion(input=logits, target=labels.float())

          loss.backward()
          optimizer.step()

        if (step + 1) % 10 == 0:
          end_time = time.time()
          logging.info(
              'Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}, Time: {:.2f}'.format(
                  epoch + 1, FLAGS.num_epochs, step + 1, len(train_loader),
                  loss.item(), (end_time - start_time) / 10))
          y_true = labels.numpy()
          y_score = logits.numpy()
          y_pred = y_score > 0.5

          logging.info("=" * 50)
          logging.info("Total: %d", y_true.shape[0])
          logging.info("Correct: %d", np.sum((y_true == y_pred)))
          logging.info(
              "Batch Accuracy: %.4f",
              sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred))
          logging.info("Batch F1 Score: %.4f",
                       sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred))
          logging.info(
              "Batch ROC AUC Score: %.4f",
              sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score))
          logging.info(
              "Batch Classification Report:\n%s",
              sklearn.metrics.classification_report(
                  y_true=y_true,
                  y_pred=y_pred,
                  target_names=["negative", "positive"]))
          logging.info("=" * 50)
          start_time = time.time()

      logging.info("Saving model checkpoint...")
      checkpoint_name = os.path.join(root_dir,
                                     "checkpoint_epoch%02d.model" % (epoch + 1))
      torch.save(model.state_dict(), checkpoint_name)

      # Resample training dataset.
      train_dataset.resample()

  except KeyboardInterrupt:
    logging.info("Interruppted. Stop training.")
    logging.info("Saving model checkpoint...")
    checkpoint_name = os.path.join(
        root_dir, "checkpoint_epoch%02d_step%03d.model" % (epoch + 1, step + 1))
    torch.save(model.state_dict(), checkpoint_name)

    logging.info("Model saved at %s", checkpoint_name)

  return


def inference(configs):
  root_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)

  for i, x in enumerate(sys.argv):
    logging.info("%d: %s", i, x)

  eval_dataset = MimicDataset(
      data_split=FLAGS.data_split,
      data_dir=configs["data_dir"],
      target_label=configs["target_label"],
      history_window=configs["history_window"],
      prediction_window=configs["prediction_window"],
      dataset_size=FLAGS.dataset_size,
      pca_dim=configs["input_size"],
      pca_decomposer_path=os.path.join(root_dir, "pca_decomposer.skmodel"),
      standardize=configs["standardize"],
      standard_scaler_path=os.path.join(root_dir, "standard_scaler.skmodel"),
  )

  eval_loader = torch.utils.data.DataLoader(
      dataset=eval_dataset,
      batch_size=FLAGS.batch_size,
      shuffle=False,
      drop_last=False,
  )

  model = RNNModel(
      input_size=configs["input_size"],
      hidden_size=configs["output_size"],
      rnn_type=configs["rnn_type"],
      num_layers=configs["rnn_layers"],
      dropout=configs["rnn_dropout"],
      bidirectional=configs["bidirectional"],
  )

  model_checkpoints = [
      x for x in os.listdir(root_dir)
      if x.endswith(".model") and "step" not in x
  ]
  logging.info("Total number of checkpoints: %d", len(model_checkpoints))
  for i, x in enumerate(model_checkpoints):
    logging.info("%d: %s", i, x)

  for checkpoint_name in model_checkpoints:
    checkpoint_path = os.path.join(root_dir, checkpoint_name)

    model.load_state_dict(torch.load(checkpoint_path))
    logging.info("Load model from: %s", checkpoint_path)

    model.eval()

    logging.info("Initializatoin complete. Start inference.")

    y_true, y_score = [], []
    with torch.set_grad_enabled(False):
      for i, (inputs, labels) in enumerate(eval_loader):
        logits, _last_embedding, _full_embedding = model(inputs)

        y_true.append(labels.numpy())
        y_score.append(logits.numpy())

        if (i + 1) % 10 == 0:
          logging.info("Progress: %d / %d", i + 1, len(eval_loader))

      y_true = np.concatenate(y_true, axis=None)
      y_score = np.concatenate(y_score, axis=None)
      y_pred = y_score > 0.5

      logging.info("=" * 50)
      logging.info("Total: %d", y_true.shape[0])
      logging.info("Correct: %d", np.sum((y_true == y_pred)))
      logging.info("Accuracy: %.4f",
                   sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred))
      logging.info("F1 Score: %.4f",
                   sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred))
      logging.info(
          "ROC AUC Score: %.4f",
          sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score))
      logging.info(
          "Classification Report:\n%s",
          sklearn.metrics.classification_report(
              y_true=y_true,
              y_pred=y_pred,
              target_names=["negative", "positive"]))
      logging.info("=" * 50)
  return


def save_and_load_flags():
  root_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
  flag_saving_path = os.path.join(root_dir, "configs.json")

  # Save model configurations
  if FLAGS.phase == "train":

    if os.path.isdir(root_dir):
      raise ValueError(
          "Target directory already exists. Please change `experiment_name`.")

    os.makedirs(root_dir)

    configs = {
        "input_size": FLAGS.input_size,
        "output_size": FLAGS.output_size,
        "rnn_type": FLAGS.rnn_type,
        "rnn_layers": FLAGS.rnn_layers,
        "rnn_dropout": FLAGS.rnn_dropout,
        "bidirectional": FLAGS.bidirectional,
        "standardize": FLAGS.standardize,
        "history_window": FLAGS.history_window,
        "prediction_window": FLAGS.prediction_window,
        "target_label": FLAGS.target_label,
        "data_dir": FLAGS.data_dir,
    }
    with open(flag_saving_path, "w") as fp:
      configs = json.dump(configs, fp, indent=2)

  assert os.path.exists(
      flag_saving_path
  ), "Training model configuration didn't find, please double check `checkpoint_dir` and `experiment_name`."

  with open(flag_saving_path, "r") as fp:
    configs = json.load(fp)

  logging.info("Saved model parameters:")
  for i, (key, val) in enumerate(configs.items()):
    logging.info("%d: %s=%s", i, key, val)

  return configs


def main(unused_argv):

  configs = save_and_load_flags()

  if FLAGS.phase == "train":
    logging.info("Mode: Training")
    train(configs)
  elif FLAGS.phase == "inference":
    logging.info("Mode: Inference")
    inference(configs)


if __name__ == "__main__":
  app.run(main)
