from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import torch


class RNNModel(torch.nn.Module):

  def __init__(self, input_size, hidden_size, rnn_type, num_layers, dropout,
               bidirectional):
    """Builds model.

    Args:
      input_size: Dimension of input vector.
      hidden_size: Dimension of hidden embeddings.
      rnn_type: `LSTM` or `GRU`.
      num_layers: Number of layers for stacked LSTM.
      dropout: Float, dropout rate.
      bidirectional: True if using bidirectional LSTM otherwise False.
    """
    super(RNNModel, self).__init__()

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

    self.rnn_module = module(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        batch_first=True,
        num_layers=self.num_layers,
        dropout=self.dropout,
        bidirectional=self.bidirectional)

    self.sigmoid = torch.nn.Linear(
        in_features=self.hidden_size * (self.bidirectional + 1), out_features=1)

    return

  def forward(self, inputs):
    """Performs forward computation.

    Args:
      inputs: (batch, sequence_length, input_size)

    Returns:
      sigmoid_logits: (batch,)
      output_embedding: (batch, num_directions * hidden_size)
      outputs: (batch, sequence_length, num_directions * hidden_size)
    """
    outputs, aux_outputs = self.rnn_module(inputs)

    if "lstm" == self.rnn_type:
      hidden_outputs, cell_outputs = aux_outputs
    elif "gru" == self.rnn_type:
      hidden_outputs = aux_outputs

    if self.bidirectional:
      (batch_size, sequence_length, dims) = list(outputs.size())
      outputs = outputs.view(batch_size, sequence_length, 2, self.hidden_size)
      forward_embedding = outputs[:, -1, 0, :]
      backward_embedding = outputs[:, 0, 1, :]
      output_embedding = torch.cat((forward_embedding, backward_embedding),
                                   dim=1)
      # logging.info("forward embedding shape: %s", forward_embedding.size())
      # logging.info("backward embedding shape: %s", backward_embedding.size())
    else:
      output_embedding = outputs[:, -1, :]

    # logging.info("output embedding shape: %s", output_embedding.size())

    sigmoid_logits = self.sigmoid(output_embedding).squeeze(1)

    return sigmoid_logits, output_embedding, outputs
