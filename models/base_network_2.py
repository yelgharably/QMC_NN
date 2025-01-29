import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network

class WavefunctionNN(nn.Module):
    def __init__(self):
        super(WavefunctionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Dropout(p=0.1),          # Added Dropout
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(p=0.1),          # Added Dropout
            nn.Linear(128, 1)
        )

    def forward(self, x):
        psi = self.fc(x)
        return psi

