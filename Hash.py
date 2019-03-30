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


# TODO placeholder for real LSTMHash function
class LSTMHash:
    def __init__(self):
        print('init')

    def hash(self, data):
        return 0


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
