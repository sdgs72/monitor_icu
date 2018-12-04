from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import os
import numpy as np
import sys
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
flags.DEFINE_integer("dataset_size", 50000,
                     "Number of instances in the each epoch.")
flags.DEFINE_boolean("standardize", True, "Whether to standardize input data.")
flags.DEFINE_string("checkpoint_dir", "./",
                    "Directory where trained models are stored.")
flags.DEFINE_string("experiment_name", None, "Identifies the experiments.")
flags.DEFINE_enum("rnn_type", "lstm", ["lstm", "gru"],
                  "Type of RNN modules for experiments.")
flags.DEFINE_integer("rnn_layers", 1, "Number of layers tacked in RNN modules.")
flags.DEFINE_float("rnn_dropout", 0.0, "Dropout rate in the RNN modules.")


def train():
  root_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
  if os.path.isdir(root_dir):
    raise ValueError(
        "Target directory already exists. Please change `experiment_name`.")
  os.makedirs(root_dir)

  with open(os.path.join(root_dir, "configs.cfg"), "w") as fp:
    fp.write("python ")
    for i, x in enumerate(sys.argv):
      logging.info("%d: %s", i, x)
      fp.write("%s \\\n" % x)

  train_dataset = MimicDataset(
      data_split=FLAGS.data_split,
      data_dir=FLAGS.data_dir,
      target_label=FLAGS.target_label,
      history_window=FLAGS.history_window,
      prediction_window=FLAGS.prediction_window,
      dataset_size=FLAGS.dataset_size,
      pca_dim=FLAGS.input_size,
      pca_decomposer_path=os.path.join(root_dir, "pca_decomposer.skmodel"),
      standardize=FLAGS.standardize,
      standard_scaler_path=os.path.join(root_dir, "standard_scaler.skmodel"),
  )

  train_loader = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=FLAGS.batch_size,
      shuffle=True,
      drop_last=False,
  )

  model = RNNModel(
      input_size=FLAGS.input_size,
      hidden_size=FLAGS.output_size,
      rnn_type=FLAGS.rnn_type,
      num_layers=FLAGS.rnn_layers,
      dropout=FLAGS.rnn_dropout,
      bidirectional=FLAGS.bidirectional,
  )

  logging.info("Training Dataset Size: %d", len(train_dataset))
  logging.info("Steps in Each Epoch: %d", len(train_loader))

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
          start_time = time.time()

      logging.info("Saving model checkpoint...")
      checkpoint_name = os.path.join(root_dir,
                                     "checkpoint_epoch%02d.model" % (epoch + 1))
      torch.save(model.state_dict(), checkpoint_name)

  except KeyboardInterrupt:
    logging.info("Interruppted. Stop training.")
    logging.info("Saving model checkpoint...")
    checkpoint_name = os.path.join(
        root_dir, "checkpoint_epoch%02d_step%03d.model" % (epoch + 1, step + 1))
    torch.save(model.state_dict(), checkpoint_name)

    logging.info("Model saved at %s", checkpoint_name)

  return


def inference():
  root_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)

  if not os.path.isdir(root_dir):
    raise ValueError(
        "Target directory doesn't exist. Please check `checkpoint_dir` and `experiment_name`."
    )

  eval_dataset = MimicDataset(
      data_split=FLAGS.data_split,
      data_dir=FLAGS.data_dir,
      target_label=FLAGS.target_label,
      history_window=FLAGS.history_window,
      prediction_window=FLAGS.prediction_window,
      dataset_size=FLAGS.dataset_size,
      pca_dim=FLAGS.input_size,
      pca_decomposer_path=os.path.join(root_dir, "pca_decomposer.skmodel"),
      standardize=FLAGS.standardize,
      standard_scaler_path=os.path.join(root_dir, "standard_scaler.skmodel"),
  )

  eval_loader = torch.utils.data.DataLoader(
      dataset=eval_dataset,
      batch_size=FLAGS.batch_size,
      shuffle=False,
      drop_last=False,
  )

  model = RNNModel(
      input_size=FLAGS.input_size,
      hidden_size=FLAGS.output_size,
      rnn_type=FLAGS.rnn_type,
      num_layers=FLAGS.rnn_layers,
      dropout=FLAGS.rnn_dropout,
      bidirectional=FLAGS.bidirectional,
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
    total, corrects = 0, 0
    with torch.set_grad_enabled(False):
      for i, (inputs, labels) in enumerate(eval_loader):
        logits, _last_embedding, _full_embedding = model(inputs)
        corrects += torch.sum((logits > 0.5).int() == labels.int()).item()
        total += labels.size(0)

        if (i + 1) % 10 == 0:
          logging.info("Progress: %d / %d", i + 1, len(eval_loader))

      logging.info("=" * 50)
      logging.info("Total: %d", total)
      logging.info("Corrects: %d", corrects)
      logging.info("Accuracy: %.4f", corrects / total)
      logging.info("=" * 50)
  return


def main(unused_argv):
  if FLAGS.phase == "train":
    logging.info("Mode: Training")
    train()
  elif FLAGS.phase == "inference":
    logging.info("Mode: Inference")
    inference()


if __name__ == "__main__":
  app.run(main)
