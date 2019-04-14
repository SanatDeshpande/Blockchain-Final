import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseHash:
    def __init__(self):
        self.model = Dense()
        self.model.load_state_dict(torch.load("./models/dense"))
        self.prev_hash = torch.zeros(256).float()

    def compute_hash(self, data):
        for i in range(0, len(data), 512):
            message = torch.cat((self.prev_hash.float(), data[i:i + 512].float()))
            self.prev_hash = self.model(message.float())
        return self.prev_hash

    def hash(self, data):
        return self.compute_hash(torch.tensor(data)).detach().numpy()


class LSTMHash:
    def __init__(self):
        self.model = LSTM(512, 256)
        self.model.load_state_dict(torch.load("./models/lstm"))
        self.hidden = (torch.zeros((1, 1, 256)), torch.zeros((1, 1, 256)))

    def compute_hash(self, data):
        output, self.hidden = self.model(data.float(), self.hidden)
        return output[0][-1] #(batches, seq_size, elements_in_one_unit) -> we only want the last one

    def hash(self, data):
        data = data.reshape(1, -1, 512)
        return self.compute_hash(torch.tensor(data)).detach().numpy()

class LSTMHashTrained:
    def __init__(self):
        self.model = LSTM(512, 256)
        self.model.load_state_dict(torch.load("./models/trained_lstm"))
        self.hidden = (torch.zeros((1, 1, 256)), torch.zeros((1, 1, 256)))

    def compute_hash(self, data):
        output, self.hidden = self.model(data.float(), self.hidden)
        return output[0][-1] #(batches, seq_size, elements_in_one_unit) -> we only want the last one

    def hash(self, data):
        data = data.reshape(1, -1, 512)
        return self.compute_hash(torch.tensor(data)).detach().numpy()

class DoubleDenseHash:
    def __init__(self):
        self.model = DoubleDense()
        self.model.load_state_dict(torch.load("./models/double_dense"))
        self.prev_hash = torch.zeros(256).float()

    def compute_hash(self, data):
        for i in range(0, len(data), 512):
            message = torch.cat((self.prev_hash.float(), data[i:i + 512].float()))
            self.prev_hash = self.model(message.float())
        return self.prev_hash

    def hash(self, data):
        return self.compute_hash(torch.tensor(data)).detach().numpy()


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
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.round(torch.sigmoid(x))


class DoubleDense(nn.Module):
    def __init__(self):
        super(DoubleDense, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        x = x * 2 - 1
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.round(torch.sigmoid(x))
