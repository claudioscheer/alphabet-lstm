import unidecode
import string
import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import numpy as np
import random


def read_whole_file(input_path):
    with open(input_path, mode="r") as file:
        return unidecode.unidecode(file.read())


class CustomDatasetLoader(Dataset):
    def __init__(self, input_path):
        self.data = read_whole_file(input_path)
        self.data_len = len(self.data)

        # Unique characters in the database.
        self.unique_characters = string.printable
        self.unique_characters_length = len(self.unique_characters)

        # Map int to character.
        self.int2char = {i: char for i, char in enumerate(self.unique_characters)}
        # Map character to int.
        self.char2int = {char: i for i, char in enumerate(self.unique_characters)}

        self.data_encoded = self.characters2int(self.data)

    def __len__(self):
        return self.data_len

    def get_data(self):
        # Remove last character.
        x = self.data_encoded[:-1]
        # Remove first character.
        y = self.data_encoded[1:]

        x = torch.tensor(x).cuda()
        y = torch.tensor(y).cuda()
        return Variable(x), Variable(y)

    def characters2int(self, characters):
        return [self.char2int[c] for c in characters]

    def int2characters(self, characters):
        return [self.int2char[c] for c in characters]
