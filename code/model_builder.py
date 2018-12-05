from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import torch


class MimicModel(torch.nn.Module):

  def __init__(self, model_type, input_size, hidden_size, rnn_type, num_layers,
               dropout, bidirectional):
    """Builds model.

    Args:
      model_type: `lr`, `rnn`, `attentional_lr` or `attentional_rnn`.
      input_size: Dimension of input vector.
      hidden_size: Dimension of hidden embeddings.
      rnn_type: `LSTM` or `GRU`.
      num_layers: Number of layers for stacked LSTM.
      dropout: Float, dropout rate.
      bidirectional: True if using bidirectional LSTM otherwise False.
    """
    super(MimicModel, self).__init__()

    self.model_type = model_type
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.rnn_type = rnn_type
    self.num_layers = num_layers
    self.dropout = dropout
    self.bidirectional = bidirectional

    if "lstm" == self.rnn_type:
      module = torch.nn.LSTM
    elif "gru" == self.rnn_type:
      module = torch.nn.GRU
    else:
      raise ValueError("Only `LSTM` and `GRU` are supported `rnn_type`.")

    if "rnn" in self.model_type:
      self.rnn_module = module(
          input_size=self.input_size,
          hidden_size=self.hidden_size,
          batch_first=True,
          num_layers=self.num_layers,
          dropout=self.dropout,
          bidirectional=self.bidirectional)

      self.rnn_linear = torch.nn.Linear(
          in_features=self.hidden_size * (self.bidirectional + 1),
          out_features=1)

      if "attentional" in self.model_type:
        self.attention_layer = torch.nn.Linear(
            in_features=self.hidden_size * (self.bidirectional + 1),
            out_features=1)

    if "lr" in self.model_type:
      self.lr_linear = torch.nn.Linear(
          in_features=self.hidden_size, out_features=1)

      if "attentional" in self.model_type:
        self.attention_layer = torch.nn.Linear(
            in_features=self.hidden_size, out_features=1)

    return

  def forward(self, inputs):
    """Performs forward computation.

    Args:
      inputs: (batch, sequence_length, input_size)

    Returns:
      logits: (batch,)
    """
    if "rnn" == self.model_type:
      return self._rnn_forward(inputs)
    elif "attentional_rnn" == self.model_type:
      return self._attentional_rnn_forward(inputs)
    elif "lr" == self.model_type:
      return self._lr_forward(inputs)
    elif "attentional_lr" == self.model_type:
      return self._attentional_lr_forward(inputs)
    else:
      raise ValueError(
          "Only `lr`, `rnn`, `attentional_lr` or `attentional_rnn` "
          "are supported `model_type`.")

  def _rnn_forward(self, inputs):
    outputs, aux_states = self.rnn_module(inputs)

    if self.bidirectional:
      (batch_size, sequence_length, _) = list(outputs.size())
      outputs = outputs.view(batch_size, sequence_length, 2, self.hidden_size)
      forward_embedding = outputs[:, -1, 0, :]
      backward_embedding = outputs[:, 0, 1, :]
      output_embedding = torch.cat((forward_embedding, backward_embedding),
                                   dim=1)
    else:
      output_embedding = outputs[:, -1, :]

    logits = self.rnn_linear(output_embedding).squeeze(1)

    return logits

  def _attentional_rnn_forward(self, inputs):
    outputs, aux_states = self.rnn_module(inputs)

    attention = self.attention_layer(outputs)
    attention_score = torch.nn.functional.softmax(attention, dim=1)

    output_embedding = torch.sum(attention_score * outputs, dim=1)
    logits = self.rnn_linear(output_embedding).squeeze(1)

    return logits

  def _lr_forward(self, inputs):
    inputs = torch.mean(inputs, dim=1)
    logits = self.lr_linear(inputs).squeeze(1)
    return logits

  def _attentional_lr_forward(self, inputs):
    attention = self.attention_layer(inputs)
    attention_score = torch.nn.functional.softmax(attention, dim=1)
    output_embedding = torch.sum(attention_score * inputs, dim=1)
    logits = self.lr_linear(output_embedding).squeeze(1)
    return logits
