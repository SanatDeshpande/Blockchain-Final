import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseHash:
    def __init__(self):
        self.model = Dense()
        self.model.load_state_dict(torch.load("./models/dense"))
        self.prev_hash = torch.zeros((256))

    def compute_hash(self, data):
        for i in range(0, len(data), 256):
            message = torch.cat((self.prev_hash, data[i:i+256]))
            self.prev_hash = self.model(message)
        return self.prev_hash

    def hash(self, data):
        return self.compute_hash(torch.tensor(data)).numpy()



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return torch.round(torch.sigmoid(output)), hidden

class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.round(torch.sigmoid(x))
