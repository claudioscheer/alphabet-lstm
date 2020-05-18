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

model = LSTMModel(dataset.unique_characters_length, dataset.unique_characters_length)
model.cuda()

print("Starting train process...")


def train_model(n_epochs, show_loss_plot=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0006)

    train_loss_over_epochs = []
    validation_loss_over_epochs = []

    for epoch in range(1, n_epochs + 1):
        hidden_states = model.init_hidden_states(1)
        optimizer.zero_grad()

        x, y = dataset.get_data()
        output, hidden_states = model(x.unsqueeze(0), hidden_states)
        train_loss = criterion(output, y)
        train_loss.backward()
        optimizer.step()

        # x_validation, y_validation = dataset.get_batch(batch_size=1, sequence_size=1)
        # hidden_states = model.init_hidden_states(1)
        # output_validation, _ = model(x_validation, hidden_states)
        # validation_loss = criterion(output_validation, y_validation.view(-1).long())

        train_loss_over_epochs.append(train_loss.item())
        # validation_loss_over_epochs.append(validation_loss.item())
        print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
        print("Loss: {:.4f}".format(train_loss.item()))

    if show_loss_plot:
        plt.plot(train_loss_over_epochs, label="Train loss")
        plt.plot(validation_loss_over_epochs, label="Validation loss")
        plt.legend()
        plt.title("Loss")
        plt.show()


train_model(n_epochs=5000, show_loss_plot=True)
torch.save(model, "../output/model.pytorch")
