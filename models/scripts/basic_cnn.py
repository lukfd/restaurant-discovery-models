import torch
from torch import nn


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.convolutional(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.lin(x)
