import torch
import torch.nn as nn
import torch.optim as optim

# New arch

class WavefunctionNN(nn.Module):
    def __init__(self):
        super(WavefunctionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 256),          # Increased from 128 to 256
            nn.Tanh(),
            nn.Dropout(p=0.2),          # Increased dropout for regularization
            
            nn.Linear(256, 256),        # Additional layer
            nn.Tanh(),
            nn.Dropout(p=0.2),
            
            nn.Linear(256, 256),        # Another additional layer
            nn.Tanh(),
            nn.Dropout(p=0.2),
            
            nn.Linear(256, 128),        # Narrowing down toward output
            nn.Tanh(),
            nn.Dropout(p=0.2),
            
            nn.Linear(128, 1),          # Final output layer
            nn.Softplus()               # Ensures non-negative outputs
        )

    def forward(self, x):
        return self.fc(x)  # No normalization in forward pass