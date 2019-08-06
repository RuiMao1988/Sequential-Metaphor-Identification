import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch


class RNNSequenceModel(nn.Module):
    # num_classes: The number of classes in the classification problem.
    # embedding_dim: The input dimension
    # hidden_size: The size of the RNN hidden state.
    # num_layers: Number of layers to use in RNN
    # bidir: boolean of wether to use bidirectional or not in RNN
    # dropout1: dropout on input to RNN
    # dropout2: dropout in RNN
    # dropout3: dropout on hidden state of RNN to linear layer
    def __init__(self, num_classes, embedding_dim, hidden_size, num_layers, bidir=True,
                 dropout1=0.2, dropout2=0.2, dropout3=0.2):

        super(RNNSequenceModel, self).__init__()

        self.rnn = nn.LSTM(input_size=embedding_dim , hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout2, batch_first=True, bidirectional=bidir)

        direc = 2 if bidir else 1

        self.output_to_label = nn.Linear(hidden_size * direc + 300, num_classes)

        self.dropout_on_input_to_LSTM = nn.Dropout(dropout1)
        self.dropout_on_input_to_linear_layer = nn.Dropout(dropout3)

        self.embedding_linear = nn.Linear(300, hidden_size * direc)
        self.tanh = nn.Tanh()

    def forward(self, inputs, lengths):

        embedded_input = self.dropout_on_input_to_LSTM(inputs)

        output, _ = self.rnn(embedded_input)

        embedding_proj = embedded_input[:,:,:300]

        output_cat = torch.cat([output, embedding_proj], -1)

        input_encoding = self.dropout_on_input_to_linear_layer(output_cat)

        unnormalized_output = self.output_to_label(input_encoding)

        output_distribution = F.log_softmax(unnormalized_output, dim=-1)

        return output_distribution
