import torch.nn as nn

class Simple1NN(nn.Module):
    def __init__(self):
        super(Simple1NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 1250),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

class Simple2NN(nn.Module):
    def __init__(self):
        super(Simple2NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 200),
            nn.ReLU(),
            nn.Linear(200, 1250),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

class Simple3NN(nn.Module):
    def __init__(self):
        super(Simple3NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 1250),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

