"""
Currently functioning properly!
Testing and Training in progress.
Something about the training isn't right.
The theta and phi coordinates were mixed up, giving out weird results
"""

import numpy as np
import torch
from ..models.base_network_2 import WavefunctionNN
import torch.optim as optim
import torch.nn as nn
from .losses_2 import QuantumLoss
from torch.utils.data import Dataset, DataLoader
from ..qmc.qmc_generator_2 import QMC_gen_v2
import matplotlib.pyplot as plt
import sys
import sympy as sp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

model_name = "model_14_tanh.pth"

class QuantumDataset(Dataset):
    def __init__(self, data, labels):
        """
        Parameters:
        - data: np.ndarray of shape (N, 4), where N is the number of samples,
                and the columns represent [r, phi, n, l].
        - labels: np.ndarray of shape (N,), representing true wavefunction values.
        """
        self.data = data  # Includes r, phi, n, l
        self.labels = labels  # True wavefunction values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "r": torch.tensor(self.data[idx, 0], dtype=torch.float32),
            "phi": torch.tensor(self.data[idx, 1], dtype=torch.float32),
            "n": torch.tensor(self.data[idx, 2], dtype=torch.float32),
            "l": torch.tensor(self.data[idx, 3], dtype=torch.float32),
            "psi_true": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
    
# Radial wavefunction R_{nl} (example for hydrogen-like atom)
def radial_wavefunction(n, l, r):
    rho = 2 * r / n
    prefactor = np.sqrt((2 / n)**3 * np.math.factorial(n - l - 1) / (2 * n * np.math.factorial(n + l)))
    laguerre = np.polynomial.laguerre.Laguerre.basis(n - l - 1)(rho)
    return prefactor * (rho**l) * np.exp(-rho / 2) * laguerre

# Angular wavefunction in the xy-plane (fix theta = pi/2)
def angular_wavefunction(l, phi, theta=np.pi / 2, m=0):
    from scipy.special import sph_harm
    return sph_harm(m, l, phi, theta)

# Full wavefunction squared
# def psi_squared(r, phi, n, l):
#     R = radial_wavefunction(n, l, r)
#     Y = angular_wavefunction(l, phi)
#     return (R * Y)**2

def psi_squared(r, phi, n, l):
    # Exact hydrogen wavefunction squared for 1s state (n=1, l=0)
    R = np.exp(-r)  # Radial part for 1s state
    Y = 1 / np.sqrt(4 * np.pi)  # Angular part (constant for l=0, m=0)
    return (R * Y)**2

class HydrogenWfc():
    def __init__(self,r,theta,n,l,m=0,phi=0):
        self.a0 = 1.0
        self.r = r
        self.theta = theta
        self.n = n
        self.l = l
        self.m = m
        self.phi = phi
    
    def Rnl(self,r,n,l):
        return radial_wavefunction(n,l,r) # n =n-l-1, alpha = 2*l+1, x = 2*r/(n*a0)
    
    def Ylm(self,theta,phi,l):
        return angular_wavefunction(l,phi)
    
    def normalized_wfc(self, r, theta, phi):
        """Return normalized wavefunction."""
        wfc = self.Rnl(r,self.n,self.l) * self.Ylm(theta,phi,self.l)


# Generate training data
def generate_data(generator, n, l, n_samples):
    samples = generator.metropolis_sampling(n_samples=n_samples, burn_in=n_samples // 5)

    r = np.linalg.norm(samples, axis=1)
    phi = np.arctan2(samples[:, 1], samples[:, 0])  # Azimuthal angle in the xy-plane
    phi = np.mod(phi, 2 * np.pi)  # Ensure phi is in [0, 2π)
    theta = np.full_like(r, np.pi / 2)  # Fix polar angle since z = 0

    n_array = np.full_like(r, n)
    l_array = np.full_like(r, l)

    psi_values = psi_squared(r, phi, n, l)  # Adjust to use phi instead of theta
    inputs = np.stack((r, phi, n_array, l_array), axis=1)

    return inputs, psi_values

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def save_no_terminal(model,path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {path}")

# Training loop
def train_model_with_quantum_loss(
    model, quantum_loss, optimizer, data_loader, save_path, epochs=100, device="cpu"
):
    try:
        load_model(model, model_name)
        print("Loaded saved model. Resuming training.")
    except FileNotFoundError:
        print("No saved model found. Starting training from scratch.")

    model.to(device)
    quantum_loss.to(device)
    losses = []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    best_loss = float('inf')
    patience_counter = 0
    patience = 100

    for epoch in range(epochs):
        model.train()
        quantum_loss.update_weights(epoch,epochs)
        epoch_loss = 0.0
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch in data_loader:
                r = batch["r"].to(device).requires_grad_(True)
                phi = batch["phi"].to(device).requires_grad_(True)
                n = batch["n"].to(device)
                l = batch["l"].to(device)
                psi_true = batch["psi_true"].to(device)

                # Stack inputs (r, theta, n, l)
                inputs = torch.stack((r, phi, n, l), dim=1)

                # Forward pass
                optimizer.zero_grad()
                psi_pred = model(inputs)

                # Compute loss (no need to pass E or V explicitly)
                loss = quantum_loss(psi_pred, r, phi, psi_true)
                epoch_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.6f}")
                pbar.update(1)

        # Store average loss for this epoch
        avg_loss = epoch_loss / len(data_loader)
        scheduler.step(avg_loss)
        save_no_terminal(model, save_path)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter=0
        else:
            patience_counter+=1
        if patience_counter>=patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        losses.append(avg_loss)
    print(f"Average Loss: {np.mean(np.array(losses)):.6f}")

    save_model(model, save_path)

    new_model = WavefunctionNN()
    load_model(new_model, model_name)

    # Check if weights match
    for param1, param2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(param1, param2), "Weights do not match!"
    print("Model saved and loaded correctly!")

    return losses

def check_data(X, y):
    assert not np.isnan(X).any(), "Input data contains NaN"
    assert not np.isinf(X).any(), "Input data contains Inf"
    assert not np.isnan(y).any(), "Target data contains NaN"
    assert not np.isinf(y).any(), "Target data contains Inf"
    print("Data integrity checks passed.")

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='tanh')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(0)
    sample_n = int(sys.argv[1])
    n = 1
    l = 0
    hwf = HydrogenWfc(r=None, theta=None, phi=None, n=n, l=l, m=0)
    epoch_count = 10

    def hydrogen_wf_wrapper(pos):
        x, y = pos[0], pos[1]
        r = np.sqrt(x**2 + y**2)  # Radial distance in 2D
        phi = np.arctan2(y, x)  # Azimuthal angle in the xy-plane
        phi = np.mod(phi, 2 * np.pi)  # Ensure phi is in [0, 2π)
        n, l = 1, 0  # Example quantum numbers
        return psi_squared(r, phi, n, l)
            
    generator = QMC_gen_v2(step_size=0.1, wavefunction=hydrogen_wf_wrapper)
    inputs, psi = generate_data(generator, n, l, sample_n)  # From earlier function
    check_data(inputs, psi)
    scaler = StandardScaler()
    inputs[:, 0] = scaler.fit_transform(inputs[:, [0]]).flatten()  # r
    inputs[:, 1] = scaler.fit_transform(inputs[:, [1]]).flatten()
    dataset = QuantumDataset(inputs, psi)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = WavefunctionNN()
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    quantum_loss = QuantumLoss(r_max=10)
    path = model_name

    losses = train_model_with_quantum_loss(
    model, quantum_loss, optimizer, data_loader, path, epochs=epoch_count, device="cuda" if torch.cuda.is_available() else "cpu")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

if __name__ == "__main__":
    main()