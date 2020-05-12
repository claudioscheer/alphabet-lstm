import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMModel, self).__init__()

        self.n_layers = 3
        self.hidden_size = 128

        self.encoder = nn.Embedding(input_size, self.hidden_size)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.n_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.dropout = nn.Dropout(0.4)
        self.decoder = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, previous_hidden_states):
        x = self.encoder(x.view(1, -1))
        output, hidden_states = self.lstm(x.view(1, 1, -1), previous_hidden_states)
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.decoder(output)
        return output, hidden_states

    def init_hidden_states(self, batch_size, use_gpu=True):
        hidden_states = (
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
        )
        if use_gpu:
            return (hidden_states[0].cuda(), hidden_states[1].cuda())
        return hidden_states
