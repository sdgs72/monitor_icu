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
#from sklearn.externals import joblib
import joblib
import time
import torch
from tensorboardX import SummaryWriter

from model_builder import MimicModel
from dataset import MimicDataset
import utilities

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "phase", "pipeline", ["train", "inference", "pipeline"],
    "Specifies whether to train a model, run inference on pretrained models, "
    "or perform evaluation on-the-fly together with training.")
flags.DEFINE_integer("batch_size", 512, "Number of instances in a batch.")
flags.DEFINE_integer("input_size", 256, "Number of features of the input.")
flags.DEFINE_integer(
    "rnn_hidden_size", 256,
    "Number of the features of the output embedding from the rnn module.")
flags.DEFINE_boolean("rnn_bidirectional", False,
                     "Whether to use bidirectional RNN model.")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs for training.")
flags.DEFINE_float("learning_rate", 1e-2, "Learning rate for model training.")
flags.DEFINE_enum("train_data_split", "train", ["train", "val", "test"],
                  "Specifies the `train`, `val` or `test` data being used.")
flags.DEFINE_enum("eval_data_split", "train", ["train", "val", "test"],
                  "Specifies the `train`, `val` or `test` data being used.")
flags.DEFINE_string("data_dir", "../data/", "Directory path to data.")
flags.DEFINE_enum("target_label", "death", ["discharge", "death", "sepsis"],
                  "Target critical event.")
flags.DEFINE_integer("history_window", 8,
                     "Number of blocks (6h/block) in history.")
flags.DEFINE_integer("prediction_window", 2,
                     "Number of blocks in future for prediction.")
flags.DEFINE_integer("block_size", 6, "Number of hours in a single block.")
flags.DEFINE_integer(
    "train_dataset_size", 0,
    "Number of instances in the each epoch. If 0 use auto-balancing mode.")
flags.DEFINE_integer(
    "eval_dataset_size", 0,
    "Number of instances in the each epoch. If 0 use auto-balancing mode.")
flags.DEFINE_boolean("standardize", True, "Whether to standardize input data.")
flags.DEFINE_string("checkpoint_dir", "./",
                    "Directory where trained models are stored.")
flags.DEFINE_string("experiment_name", None, "Identifies the experiments.")
flags.DEFINE_enum("rnn_type", "gru", ["lstm", "gru"],
                  "Type of RNN modules for experiments.")
flags.DEFINE_integer("rnn_layers", 1, "Number of layers tacked in RNN modules.")
flags.DEFINE_float("rnn_dropout", 0.0, "Dropout rate in the RNN modules.")
flags.DEFINE_enum("model_type", "rnn", ["lr", "rnn"],
                  "Type of model used for experiments.")
flags.DEFINE_boolean("use_attention", False,
                     "Whether to use attention mechanism or not.")
flags.DEFINE_enum("lr_pooling", "mean", ["mean", "max", "last", "concat"],
                  "Specifies pooling strategies for logistic regression.")
flags.DEFINE_integer(
    "save_per_epochs", 10,
    "Save intermediate checkpoints every few training epochs.")
flags.DEFINE_string("eval_checkpoint", None,
                    "Specifies a checkpoint for inference.")
flags.DEFINE_integer(
    "upper_bound_factor", 5,
    "upper bound factor to reduce oversampling negatives for long admission")
flags.DEFINE_integer(
    "fix_eval_dataset_seed", None,
    "Whether to fix the generated dataset (seed for random generator).")

default_decomposer_name = "pca_decomposer.joblib"
default_standard_scaler_name = "standard_scaler.joblib"


def train(configs):
  root_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
  phase = "training"

  with open(os.path.join(root_dir, "running_configs.cfg"), "w") as fp:
    fp.write("python ")
    for i, x in enumerate(sys.argv):
      logging.info("%d: %s", i, x)
      fp.write("%s \\\n" % x)

  logging.info("Creating dataset...")
  train_dataset = MimicDataset(
      data_split=FLAGS.train_data_split,
      data_dir=configs["data_dir"],
      block_size=configs["block_size"],
      target_label=configs["target_label"],
      history_window=configs["history_window"],
      prediction_window=configs["prediction_window"],
      dataset_size=FLAGS.train_dataset_size,
      pca_dim=configs["input_size"],
      pca_decomposer_path=os.path.join(root_dir, default_decomposer_name),
      standardize=configs["standardize"],
      standard_scaler_path=os.path.join(root_dir, default_standard_scaler_name),
      phase="training",
      upper_bound_factor=FLAGS.upper_bound_factor,
  )
  logging.info("Creating dataset completed!")

  logging.info("Creating dataset loader...")
  train_loader = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=FLAGS.batch_size,
      shuffle=True,
      drop_last=False,
  )
  logging.info("Creating dataset loader completed!")

  logging.info("Creating model for training...")
  model = MimicModel(
      model_type=configs["model_type"],
      input_size=configs["input_size"],
      use_attention=configs["use_attention"],
      rnn_hidden_size=configs["rnn_hidden_size"],
      rnn_type=configs["rnn_type"],
      rnn_layers=configs["rnn_layers"],
      rnn_dropout=configs["rnn_dropout"],
      rnn_bidirectional=configs["rnn_bidirectional"],
      lr_pooling=configs["lr_pooling"],
      lr_history_window=configs["history_window"],
  )
  logging.info("Creating model for training completed.")

  logging.info("Training dataset size: %d", len(train_dataset))
  logging.info("Batches in each epoch: %d", len(train_loader))

  model.train()

  criterion = torch.nn.BCEWithLogitsLoss()

  trainable_params = [x for x in model.named_parameters() if x[1].requires_grad]

  logging.info("Model parameters:")
  for name, param in trainable_params:
    logging.info("%s: %s", name, param.size())

  optimizer = torch.optim.RMSprop(model.parameters(), lr=FLAGS.learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer, step_size=FLAGS.num_epochs // 5, gamma=0.3)

  summary_writer = SummaryWriter(log_dir=root_dir)

  logging.info("Initializatoin complete. Start training.")
  try:
    for epoch in range(FLAGS.num_epochs):
      scheduler.step()

      summary_writer.add_scalar("learning_rate",
                                scheduler.get_lr()[0], epoch + 1)

      # Resample training dataset.
      train_dataset.resample()
      y_true, y_score = [], []

      start_time = time.time()
      for step, (inputs, labels, _) in enumerate(train_loader):

        logging.info("Data shape: %s", inputs.shape)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
          logits, _ = model(inputs)
          loss = criterion(input=logits, target=labels.float())
          summary_writer.add_scalar("training/loss", loss.item(),
                                    epoch * len(train_loader) + step)
          loss.backward()
          optimizer.step()

        y_true.append(labels.detach().numpy())
        y_score.append(logits.detach().numpy())

        if (step + 1) % 10 == 0:
          end_time = time.time()
          logging.info(
              'Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}, Speed: {:.4f} ms'.
              format(epoch + 1, FLAGS.num_epochs, step + 1, len(train_loader),
                     loss.item(),
                     (end_time - start_time) * 100 / logits.shape[0]))
          start_time = time.time()

      utilities.update_metrics(y_true, y_score, phase, summary_writer,
                               epoch + 1)

      if (epoch + 1) % configs["save_per_epochs"] == 0:
        logging.info("Saving model checkpoint...")
        checkpoint_name = os.path.join(
            root_dir, "checkpoint_epoch%03d.model" % (epoch + 1))
        torch.save(model.state_dict(), checkpoint_name)

  except KeyboardInterrupt:
    logging.info("Interruppted. Stop training.")
    logging.info("Saving model checkpoint...")
    checkpoint_name = os.path.join(
        root_dir, "checkpoint_epoch%03d_step%03d.model" % (epoch + 1, step + 1))
    torch.save(model.state_dict(), checkpoint_name)

    logging.info("Model saved at %s", checkpoint_name)
  finally:
    logging.info("Training is terminated.")

  return


def inference(configs):
  root_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
  phase = "evaluation"

  for i, x in enumerate(sys.argv):
    logging.info("%d: %s", i, x)

  eval_dataset = MimicDataset(
      data_split=FLAGS.eval_data_split,
      data_dir=configs["data_dir"],
      block_size=configs["block_size"],
      target_label=configs["target_label"],
      history_window=configs["history_window"],
      prediction_window=configs["prediction_window"],
      dataset_size=FLAGS.eval_dataset_size,
      pca_dim=configs["input_size"],
      pca_decomposer_path=os.path.join(root_dir, default_decomposer_name),
      standardize=configs["standardize"],
      standard_scaler_path=os.path.join(root_dir, default_standard_scaler_name),
      phase="inference",
      upper_bound_factor=configs["upper_bound_factor"],
      fix_dataset_seed=configs["fix_eval_dataset_seed"],
  )

  eval_loader = torch.utils.data.DataLoader(
      dataset=eval_dataset,
      batch_size=FLAGS.batch_size,
      shuffle=True,
      drop_last=False,
  )

  model = MimicModel(
      model_type=configs["model_type"],
      input_size=configs["input_size"],
      use_attention=configs["use_attention"],
      rnn_hidden_size=configs["rnn_hidden_size"],
      rnn_type=configs["rnn_type"],
      rnn_layers=configs["rnn_layers"],
      rnn_dropout=configs["rnn_dropout"],
      rnn_bidirectional=configs["rnn_bidirectional"],
      lr_pooling=configs["lr_pooling"],
      lr_history_window=configs["history_window"],
  )

  if FLAGS.eval_checkpoint:
    logging.info("Evaluate checkpoint %s", FLAGS.eval_checkpoint)
    model_checkpoints = [FLAGS.eval_checkpoint]
  else:
    model_checkpoints = [
        os.path.join(root_dir, x)
        for x in os.listdir(root_dir)
        if x.endswith(".model") and "step" not in x
    ]
    logging.info("Total number of checkpoints: %d", len(model_checkpoints))
    for i, x in enumerate(model_checkpoints):
      logging.info("%d: %s", i, x)

  for checkpoint_path in model_checkpoints:
    model.load_state_dict(torch.load(checkpoint_path))
    logging.info("Load model from: %s", checkpoint_path)

    prediction = utilities.Prediction(
        block_size=configs["block_size"],
        history_window=configs["history_window"],
        prediction_window=configs["prediction_window"])

    model.eval()

    logging.info("Initializatoin complete. Start inference.")

    y_true, y_score = [], []
    with torch.set_grad_enabled(False):
      for i, (inputs, labels, data_info) in enumerate(eval_loader):
        logits, endpoints = model(inputs)

        if "attention_scores" in endpoints:
          attention_score = endpoints["attention_scores"]
        else:
          attention_score = None

        if "outputs" in endpoints:
          outputs = endpoints["outputs"]
        else:
          outputs = None

        prediction.add_prediction(data_info, logits, labels, attention_score,
                                  outputs)

        y_true.append(labels.numpy())
        y_score.append(logits.numpy())

        if (i + 1) % 10 == 0:
          logging.info("Progress: %d / %d", i + 1, len(eval_loader))

      utilities.update_metrics(y_true, y_score, phase)

    prediction.save_inference_results(
        checkpoint_path.replace(".model",
                                "_eval_on_%s.joblib" % FLAGS.eval_data_split))
  logging.info("Evaluation on %d models complete.", len(model_checkpoints))

  return


def pipeline(configs):
  root_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)

  with open(os.path.join(root_dir, "running_configs.cfg"), "w") as fp:
    fp.write("python ")
    for i, x in enumerate(sys.argv):
      logging.info("%d: %s", i, x)
      fp.write("%s \\\n" % x)

  train_dataset = MimicDataset(
      data_split=FLAGS.train_data_split,
      data_dir=configs["data_dir"],
      block_size=configs["block_size"],
      target_label=configs["target_label"],
      history_window=configs["history_window"],
      prediction_window=configs["prediction_window"],
      dataset_size=FLAGS.train_dataset_size,
      pca_dim=configs["input_size"],
      pca_decomposer_path=os.path.join(root_dir, default_decomposer_name),
      standardize=configs["standardize"],
      standard_scaler_path=os.path.join(root_dir, default_standard_scaler_name),
      phase="training",
      upper_bound_factor=configs["upper_bound_factor"],
  )
  train_loader = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=FLAGS.batch_size,
      shuffle=True,
      drop_last=False,
  )

  eval_dataset = MimicDataset(
      data_split=FLAGS.eval_data_split,
      data_dir=configs["data_dir"],
      block_size=configs["block_size"],
      target_label=configs["target_label"],
      history_window=configs["history_window"],
      prediction_window=configs["prediction_window"],
      dataset_size=FLAGS.eval_dataset_size,
      pca_dim=configs["input_size"],
      pca_decomposer_path=os.path.join(root_dir, default_decomposer_name),
      standardize=configs["standardize"],
      standard_scaler_path=os.path.join(root_dir, default_standard_scaler_name),
      phase="inference",
      upper_bound_factor=configs["upper_bound_factor"],
      fix_dataset_seed=configs["fix_eval_dataset_seed"],
  )
  eval_loader = torch.utils.data.DataLoader(
      dataset=eval_dataset,
      batch_size=FLAGS.batch_size,
      shuffle=True,
      drop_last=False,
  )

  model = MimicModel(
      model_type=configs["model_type"],
      input_size=configs["input_size"],
      use_attention=configs["use_attention"],
      rnn_hidden_size=configs["rnn_hidden_size"],
      rnn_type=configs["rnn_type"],
      rnn_layers=configs["rnn_layers"],
      rnn_dropout=configs["rnn_dropout"],
      rnn_bidirectional=configs["rnn_bidirectional"],
      lr_pooling=configs["lr_pooling"],
      lr_history_window=configs["history_window"],
  )

  logging.info("Training dataset size: %d", len(train_dataset))
  logging.info("Batches in each epoch: %d", len(train_loader))
  logging.info("Evaluation on `%s`", FLAGS.eval_data_split)
  logging.info("Evaluation dataset size: %d", len(eval_dataset))
  logging.info("Batches in each epoch: %d", len(eval_loader))

  criterion = torch.nn.BCEWithLogitsLoss()

  trainable_params = [x for x in model.named_parameters() if x[1].requires_grad]

  logging.info("Model parameters:")
  for name, param in trainable_params:
    logging.info("%s: %s", name, param.size())

  optimizer = torch.optim.RMSprop(model.parameters(), lr=FLAGS.learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer, step_size=FLAGS.num_epochs // 5, gamma=0.3)

  best_metrics = {}
  best_checkpoint_name = os.path.join(
      root_dir, "checkpoint_best_{metric}_on_%s_epoch{epoch:03d}.model"
  ) % FLAGS.eval_data_split

  summary_writer = SummaryWriter(log_dir=root_dir)

  try:
    for epoch in range(FLAGS.num_epochs):
      logging.info("Start training.")
      phase = "training"
      model.train()

      scheduler.step()
      summary_writer.add_scalar("learning_rate",
                                scheduler.get_lr()[0], epoch + 1)

      # Resample training dataset.
      train_dataset.resample()
      y_true, y_score = [], []

      start_time = time.time()
      for step, (inputs, labels, _) in enumerate(train_loader):

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
          logits, _ = model(inputs)
          loss = criterion(input=logits, target=labels.float())
          summary_writer.add_scalar("%s/loss" % phase, loss.item(),
                                    epoch * len(train_loader) + step)
          loss.backward()
          optimizer.step()

        y_true.append(labels.detach().numpy())
        y_score.append(logits.detach().numpy())

        if (step + 1) % 10 == 0:
          end_time = time.time()
          logging.info(
              'Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}, Speed: {:.4f} ms'.
              format(epoch + 1, FLAGS.num_epochs, step + 1, len(train_loader),
                     loss.item(),
                     (end_time - start_time) * 100 / logits.shape[0]))
          start_time = time.time()

      utilities.update_metrics(y_true, y_score, phase, summary_writer,
                               epoch + 1)

      checkpoint_name = os.path.join(root_dir,
                                     "checkpoint_epoch%03d.model" % (epoch + 1))

      if (epoch + 1) % configs["save_per_epochs"] == 0:
        logging.info("Saving model checkpoint...")
        torch.save(model.state_dict(), checkpoint_name)

      phase = "evaluation"
      model.eval()
      logging.info("Start inference.")

      y_true, y_score = [], []
      with torch.set_grad_enabled(False):
        for i, (inputs, labels, _) in enumerate(eval_loader):
          logits, endpoints = model(inputs)
          loss = criterion(input=logits, target=labels.float())

          y_true.append(labels.numpy())
          y_score.append(logits.numpy())

          if (i + 1) % 10 == 0:
            logging.info("Progress: %d / %d", i + 1, len(eval_loader))

      accuracy, f1, roc_auc, ap, pr_auc = utilities.update_metrics(
          y_true, y_score, phase, summary_writer, epoch + 1)

      metrics = {
          "accuracy": accuracy,
          "f1": f1,
          "roc_auc": roc_auc,
          "ap": ap,
          "pr_auc": pr_auc,
      }

      for metric_name, metric in metrics.items():
        if (metric_name not in best_metrics or
            metric >= best_metrics[metric_name][0]):
          if metric_name in best_metrics:
            old_checkpoint_name = best_checkpoint_name.format(
                metric=metric_name, epoch=best_metrics[metric_name][1])
            if os.path.exists(old_checkpoint_name):
              os.remove(old_checkpoint_name)

          torch.save(
              model.state_dict(),
              best_checkpoint_name.format(metric=metric_name, epoch=epoch + 1))
          best_metrics[metric_name] = (metric, epoch + 1)
          logging.info("Saving best model: best %s of %.4f in epoch %d",
                       metric_name, metric, epoch + 1)

  except KeyboardInterrupt:
    logging.info("Interruppted. Stop training.")

    logging.info("Saving model checkpoint...")
    checkpoint_name = os.path.join(
        root_dir, "checkpoint_epoch%03d_step%03d.model" % (epoch + 1, step + 1))
    torch.save(model.state_dict(), checkpoint_name)
    logging.info("Model saved at %s", checkpoint_name)

  return


def save_and_load_flags():
  root_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
  flag_saving_path = os.path.join(root_dir, "configs.json")

  # Save model configurations
  if FLAGS.phase == "train" or FLAGS.phase == "pipeline":

    if os.path.isdir(root_dir):
      raise ValueError(
          "Target directory already exists. Please change `experiment_name`.")

    os.makedirs(root_dir)

    configs = {
        "model_type": FLAGS.model_type,
        "input_size": FLAGS.input_size,
        "rnn_hidden_size": FLAGS.rnn_hidden_size,
        "use_attention": FLAGS.use_attention,
        "rnn_type": FLAGS.rnn_type,
        "rnn_layers": FLAGS.rnn_layers,
        "rnn_dropout": FLAGS.rnn_dropout,
        "rnn_bidirectional": FLAGS.rnn_bidirectional,
        "standardize": FLAGS.standardize,
        "block_size": FLAGS.block_size,
        "history_window": FLAGS.history_window,
        "prediction_window": FLAGS.prediction_window,
        "target_label": FLAGS.target_label,
        "data_dir": FLAGS.data_dir,
        "lr_pooling": FLAGS.lr_pooling,
        "save_per_epochs": FLAGS.save_per_epochs,
        "upper_bound_factor": FLAGS.upper_bound_factor,
        "fix_eval_dataset_seed": FLAGS.fix_eval_dataset_seed,
    }

    with open(flag_saving_path, "w") as fp:
      configs = json.dump(configs, fp, indent=2)

  if not os.path.exists(flag_saving_path):
    raise AssertionError(
        "Training model configuration didn't find, please double check "
        "`checkpoint_dir` and `experiment_name`.")

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
  elif FLAGS.phase == "pipeline":
    logging.info("Mode: Train/Eval pipeline")
    pipeline(configs)


if __name__ == "__main__":
  app.run(main)
