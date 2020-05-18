import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.dataset import CustomDatasetLoader
from lib.model import LSTMModel
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))
dataset = CustomDatasetLoader(os.path.join(dir_path, "../dataset/alphabet.txt"))

model = torch.load(os.path.join(dir_path, "../output/model.pytorch"))
model.cpu()
model.eval()


def evaluate(model, start_text, prediction_length):
    predicted_alphabet = start_text
    size_prediction = prediction_length - len(start_text)

    # Convert start_text to an array of integers.
    start_text = Variable(torch.tensor(dataset.characters2int(start_text)))

    # Init hidden state with zeros.
    previous_hidden_states = model.init_hidden_states(1, use_gpu=False)
    # -1 because the next for loop will perform the prediction based on the last character.
    for c in range(len(start_text) - 1):
        _, previous_hidden_states = model(
            start_text[c].unsqueeze(0).unsqueeze(0).long(), previous_hidden_states
        )

    current_character = start_text[-1]

    for _ in range(size_prediction):
        output, previous_hidden_states = model(
            current_character.unsqueeze(0).unsqueeze(0).long(), previous_hidden_states
        )

        output = nn.functional.softmax(output.view(-1), dim=0).data
        max_output_index = np.argmax(output)

        predicted_char = dataset.int2char[max_output_index.item()]
        predicted_alphabet += predicted_char
        current_character = Variable(torch.tensor(dataset.char2int[predicted_char]))

    return predicted_alphabet


with torch.no_grad():
    prediction = evaluate(model, "b", 20)
    print("".join(prediction))
