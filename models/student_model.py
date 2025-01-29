from ..models.base_network_3 import WavefunctionNN
from ..training.trainer_3 import load_model, model_name
import torch
import numpy as np
from sympy import symbols, tanh, Matrix, sqrt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

student_model_name = 'student_model_009.pth'

class StudentWavefunctionNN(nn.Module):
    def __init__(self):
        super(StudentWavefunctionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),          # Reduce input-to-hidden size
            nn.Tanh(),
            nn.Linear(64, 32),         # Reduce hidden-to-hidden size
            nn.Tanh(),
            nn.Linear(32, 1)           # Single output (psi)
        )

    def forward(self, x):
        return self.fc(x)

x_data = np.linspace(-3, 3, 300)
y_data = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x_data, y_data)
X = X.flatten()
Y = Y.flatten()
n = 1 
l = 0 

inputs = torch.tensor(np.column_stack([X, Y, np.full_like(X, n), np.full_like(Y, l)]), dtype=torch.float32)

teacher_model = WavefunctionNN()
teacher_model.load_state_dict(
    torch.load("model_009_tanh.pth", map_location=torch.device('cpu'))
)
teacher_model.eval()

with torch.no_grad():
    teacher_outputs = teacher_model(inputs)

student_model = StudentWavefunctionNN()
criterion = nn.MSELoss()  # Mean Squared Error to match teacher outputs
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

try:
    load_model(student_model, student_model_name)
    print("Loaded saved model. Resuming training.")
except FileNotFoundError:
    print("No saved model found. Starting training from scratch.")

epochs = 1000
batch_size = 64
inputs = inputs.to('cuda') if torch.cuda.is_available() else inputs
teacher_outputs = teacher_outputs.to('cuda') if torch.cuda.is_available() else teacher_outputs
student_model = student_model.to('cuda') if torch.cuda.is_available() else student_model

for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
    epoch_loss = 0  # Track loss per epoch
    for i in range(0, len(inputs), batch_size):
        batch_x = inputs[i:i+batch_size]
        batch_y = teacher_outputs[i:i+batch_size]

        # Forward pass
        outputs = student_model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # Accumulate loss

    # Optionally, print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

student_model.eval()
with torch.no_grad():
    student_outputs = student_model(inputs)

torch.save(student_model.state_dict(), student_model_name)

# Compare teacher vs. student predictions
import matplotlib.pyplot as plt

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plt.figure(figsize=(10, 6))
plt.scatter(X, teacher_outputs.cpu().numpy(), label="Teacher Outputs", s=5)
plt.scatter(X, student_outputs.cpu().numpy(), label="Student Outputs", s=5, alpha=0.7)
plt.legend()
plt.xlabel("Input (x, y)")
plt.ylabel("psi")
plt.title("Comparison of Teacher and Student Models")
plt.savefig(f'Teacher_Student_{timestamp}.png')

