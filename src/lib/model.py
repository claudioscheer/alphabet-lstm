import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, bidirectional=False):
        super(LSTMModel, self).__init__()

        self.n_layers = 1
        self.hidden_size = 32
        self.bidirectional = bidirectional

        self.encoder = nn.Embedding(input_size, self.hidden_size)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if bidirectional:
            self.decoder = nn.Linear(self.hidden_size * 2, output_size)
        else:
            self.decoder = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, previous_hidden_states):
        x = self.encoder(x)
        output, hidden_states = self.lstm(x, previous_hidden_states)
        if self.bidirectional:
            output = output.contiguous().view(-1, self.hidden_size * 2)
        else:
            output = output.contiguous().view(-1, self.hidden_size)
        output = self.decoder(output)
        return output, hidden_states

    def init_hidden_states(self, batch_size, use_gpu=True):
        n_layers = self.n_layers
        if self.bidirectional:
            n_layers *= 2
        hidden_states = (
            Variable(torch.zeros(n_layers, batch_size, self.hidden_size)),
            Variable(torch.zeros(n_layers, batch_size, self.hidden_size)),
        )
        if use_gpu:
            return (hidden_states[0].cuda(), hidden_states[1].cuda())
        return hidden_states
