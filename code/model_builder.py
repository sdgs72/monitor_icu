from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import torch


class MimicModel(torch.nn.Module):

  def __init__(
      self,
      model_type,
      input_size,
      use_attention,
      rnn_hidden_size,
      rnn_type,
      rnn_layers,
      rnn_dropout,
      rnn_bidirectional,
      # train_embedding,
      # vocabulary_path,
      lr_pooling="mean",
      lr_history_window=None,
  ):
    """Builds model.

    Args:
      model_type: `lr` or `rnn`.
      input_size: Dimension of input vector.
      rnn_hidden_size: Dimension of hidden embeddings.
      rnn_type: `LSTM` or `GRU`.
      num_layers: Number of layers for stacked LSTM.
      dropout: Float, dropout rate.
      rnn_bidirectional: True if using bidirectional LSTM otherwise False.
      use_attention: True if using attention mechanism otherwise False.
      lr_pooling: `concat`, `last`, `mean` or `max`.
    """
    super(MimicModel, self).__init__()

    self.model_type = model_type
    self.input_size = input_size
    self.rnn_hidden_size = rnn_hidden_size
    self.rnn_type = rnn_type
    self.rnn_layers = rnn_layers
    self.rnn_dropout = rnn_dropout
    self.rnn_bidirectional = rnn_bidirectional
    self.use_attention = use_attention
    self.lr_pooling = lr_pooling
    self.lr_history_window = lr_history_window
    # self.train_embedding = train_embedding

    if self.rnn_type == "lstm":
      module = torch.nn.LSTM
    elif self.rnn_type == "gru":
      module = torch.nn.GRU
    else:
      raise ValueError("Only `LSTM` and `GRU` are supported `rnn_type`.")

    if self.model_type == "rnn" and self.rnn_bidirectional:
      self.rnn_hidden_size /= 2

    # if self.train_embedding:
    #   if not vocabulary_path or not os.path.exists(vocabulary_path):
    #     raise AssertionError("`vocabulary_path` is not defined or not found.")

    #   self.vocabulary = self.load_vocabulary(vocabulary_path)
    #   self.embedding = torch.nn.Embedding(len(self.vocabulary), self.input_size)

    if self.model_type == "rnn":
      self.rnn_module = module(
          input_size=self.input_size,
          hidden_size=self.rnn_hidden_size,
          batch_first=True,
          num_layers=self.rnn_layers,
          dropout=self.rnn_dropout,
          bidirectional=self.rnn_bidirectional)

      self.rnn_linear = torch.nn.Linear(
          in_features=self.rnn_hidden_size * (self.rnn_bidirectional + 1),
          out_features=1)

      if self.use_attention:
        self.attention_layer = torch.nn.Linear(
            in_features=self.rnn_hidden_size * (self.rnn_bidirectional + 1),
            out_features=1)

    if self.model_type == "lr":
      if self.use_attention:
        self.attention_layer = torch.nn.Linear(
            in_features=self.input_size, out_features=1)

        self.lr_linear = torch.nn.Linear(
            in_features=self.input_size, out_features=1)
      elif self.lr_pooling == "concat":
        self.lr_linear = torch.nn.Linear(
            in_features=self.input_size * self.lr_history_window,
            out_features=1)
      else:
        self.lr_linear = torch.nn.Linear(
            in_features=self.input_size, out_features=1)
    return

  def forward(self, inputs):
    """Performs forward computation.

    Args:
      inputs: (batch, sequence_length, input_size)

    Returns:
      logits: (batch,)
      endpoints: Dictionary of auxiliary information.
    """
    if self.model_type == "rnn":
      if self.use_attention:
        return self._attentional_rnn_forward(inputs)
      else:
        return self._rnn_forward(inputs)
    elif self.model_type == "lr":
      if self.use_attention:
        return self._attentional_lr_forward(inputs)
      else:
        return self._lr_baseline_forward(inputs)
    else:
      raise ValueError("Only `lr` and `rnn` are supported `model_type`.")

  def _rnn_forward(self, inputs):
    outputs, aux_states = self.rnn_module(inputs)
    output_embedding = aux_states[0]

    if self.rnn_bidirectional:
      output_embedding = output_embedding.transpose(0, 1).reshape(
          batch_size, self.rnn_hidden_size * 2)
    else:
      output_embedding = output_embedding.squeeze(dim=0)

    logits = self.rnn_linear(output_embedding).squeeze(1)

    endpoints = {"outputs": outputs, "aux_states": aux_states}

    return logits, endpoints

  def _attentional_rnn_forward(self, inputs):
    outputs, aux_states = self.rnn_module(inputs)

    attention = self.attention_layer(outputs)
    attention_score = torch.nn.functional.softmax(attention, dim=1)

    output_embedding = torch.sum(attention_score * outputs, dim=1)
    logits = self.rnn_linear(output_embedding).squeeze(1)

    endpoints = {
        "attention_scores": attention_score,
        "outputs": outputs,
        "aux_states": aux_states,
    }

    return logits, endpoints

  def _lr_baseline_forward(self, inputs):
    if self.lr_pooling == "mean":
      inputs = torch.mean(inputs, dim=1)
    elif self.lr_pooling == "last":
      inputs = inputs[:, -1, :]
    elif self.lr_pooling == "concat":
      batch_size = inputs.shape[0]
      inputs = torch.reshape(inputs, [batch_size, -1])
    elif self.lr_pooling == "max":
      inputs, _ = torch.max(inputs, dim=1)
    else:
      raise ValueError(
          "Only `mean`, `last`, `concat` and `max` are supported `lr_pooling`.")

    logits = self.lr_linear(inputs).squeeze(1)

    return logits, {}

  def _attentional_lr_forward(self, inputs):
    attention = self.attention_layer(inputs)
    attention_score = torch.nn.functional.softmax(attention, dim=1)
    output_embedding = torch.sum(attention_score * inputs, dim=1)
    logits = self.lr_linear(output_embedding).squeeze(1)
    endpoints = {"attention_score": attention_score}
    return logits, endpoints
