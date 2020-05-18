import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from lib.dataset import CustomDatasetLoader
from lib.model import LSTMModel


torch.autograd.set_detect_anomaly(True)

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset = CustomDatasetLoader(os.path.join(dir_path, "../dataset/alphabet.txt"))

bidirectional = False
model = LSTMModel(
    dataset.unique_characters_length, dataset.unique_characters_length, bidirectional=bidirectional
)
model.cuda()

print("Starting train process...")


def train_model(n_epochs, show_loss_plot=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    train_loss_over_epochs = []
    # validation_loss_over_epochs = []

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        x, y = dataset.get_data(bidirectional)
        hidden_states = model.init_hidden_states(batch_size=1)
        output, _ = model(x.unsqueeze(0), hidden_states)
        train_loss = criterion(output, y)
        train_loss.backward()
        optimizer.step()

        # This validation is wrong, because I am using the same data that I am using to train the model.
        # x_validation, y_validation = dataset.get_validation_data(sequence_lenght=5)
        # hidden_states = model.init_hidden_states(batch_size=1)
        # output_validation, _ = model(x_validation.unsqueeze(0), hidden_states)
        # validation_loss = criterion(output_validation, y_validation)

        train_loss_over_epochs.append(train_loss.item())
        # validation_loss_over_epochs.append(validation_loss.item())
        print("Epoch: {}/{}...".format(epoch, n_epochs), end=" ")
        print("Train Loss: {:.4f}".format(train_loss.item()))
        # print("Val. Loss: {:.4f}".format(validation_loss.item()))

    if show_loss_plot:
        plt.plot(train_loss_over_epochs, label="Train Loss")
        # plt.plot(validation_loss_over_epochs, label="Validation Loss")
        plt.legend()
        plt.title("Loss")
        plt.show()


train_model(n_epochs=500, show_loss_plot=True)
torch.save(model, "../output/model.pytorch")
