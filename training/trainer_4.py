"""
Currently functioning properly!
Testing and Training in progress.
Something about the training isn't right.
The theta and phi coordinates were mixed up, giving out weird results
"""

import numpy as np
import torch
from ..models.base_network_4_rcs import WavefunctionNN
import torch.optim as optim
import torch.nn as nn
from .losses_4 import QuantumLoss_XYZ
from torch.utils.data import Dataset, DataLoader
from ..qmc.qmc_generator_4 import QMC_gen_v4
import matplotlib.pyplot as plt
import sys
import sympy as sp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from scipy.special import factorial

model_name = "model_015_tanh.pth"

class QuantumDataset(Dataset):
    def __init__(self, data, labels):
        """
        Parameters:
        - data: np.ndarray of shape (N, 5), where N is the number of samples,
                and the columns represent [x, y, n, l, m].
        - labels: np.ndarray of shape (N, 2), representing true wavefunction values
                  [psi_real, psi_imag].
        """
        self.data = data  # Includes x, y, n, l, m
        self.labels = labels  # True wavefunction values (real and imaginary parts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.data[idx, 0], dtype=torch.float32),
            "y": torch.tensor(self.data[idx, 1], dtype=torch.float32),
            "n": torch.tensor(self.data[idx, 2], dtype=torch.long),  # No change
            "l": torch.tensor(self.data[idx, 3], dtype=torch.long),  # No change
            "m": torch.tensor(self.data[idx, 4], dtype=torch.long),
            "psi_true_real": torch.tensor(self.labels[idx, 0], dtype=torch.float32),
            "psi_true_imag": torch.tensor(self.labels[idx, 1], dtype=torch.float32),
        }

# Radial wavefunction R_{nl} (example for hydrogen-like atom)
def radial_wavefunction(n, l, r):
    # Ensure inputs are arrays for vectorized or iterative processing
    n = np.atleast_1d(n)
    l = np.atleast_1d(l)
    r = np.atleast_1d(r)
    
    # Output container for results
    result = np.zeros_like(r, dtype=np.float64)
    
    for i in range(len(n)):
        if n[i] - l[i] - 1 < 0:
            raise ValueError(f"Invalid quantum numbers: n={n[i]}, l={l[i]}")
        rho = 2 * r / n[i]
        prefactor = np.sqrt((2 / n[i])**3 * np.math.factorial(n[i] - l[i] - 1) / (2 * n[i] * np.math.factorial(n[i] + l[i])))
        laguerre = np.polynomial.laguerre.Laguerre.basis(n[i] - l[i] - 1)(rho)
        result += prefactor * (rho**l[i]) * np.exp(-rho / 2) * laguerre  # Accumulate the result
    return result


from scipy.special import sph_harm
def angular_wavefunction(l, m, theta, phi):
    """
    Computes the real and imaginary parts of the spherical harmonics.

    Parameters:
        l, m: Arrays of quantum numbers.
        theta, phi: Arrays of angular coordinates.

    Returns:
        Y_real, Y_imag: Arrays of the real and imaginary parts of spherical harmonics.
    """
    # Ensure inputs are arrays
    l = np.atleast_1d(l)
    m = np.atleast_1d(m)

    # Validate quantum numbers
    if np.any(abs(m) > l) or np.any(l < 0):
        raise ValueError(f"Invalid quantum numbers: l={l}, m={m}")

    # Compute spherical harmonics (vectorized)
    Y = sph_harm(m[:, None], l[:, None], phi, theta)  # Broadcasting across inputs

    # Extract real and imaginary parts
    Y_real = Y.real
    Y_imag = Y.imag

    return Y_real, Y_imag

# Full wavefunction squared
def psi_squared(x, y, n, l,m):
    r = np.sqrt(x**2 + y**2)
    R = radial_wavefunction(n, l, r)
    theta = np.full_like(r,np.pi/2)
    phi = np.arctan2(y,x)
    Y_real,Y_imag = angular_wavefunction(l,m,phi,theta)
    return (R*Y_real)**2 + (R*Y_imag)**2

def psi_func(x, y, n, l, m):
    """
    Computes the real and imaginary parts of the wavefunction for scalar or array inputs.
    """
    # Ensure inputs are arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    n = np.atleast_1d(n)
    l = np.atleast_1d(l)
    m = np.atleast_1d(m)

    # Compute r, theta, phi
    r = np.sqrt(x**2 + y**2).reshape(-1)  # Shape (109,)
    theta = np.full_like(r, np.pi / 2)  # Shape (109,)
    phi = np.arctan2(y, x).reshape(-1)  # Shape (109,)

    # Initialize wavefunction outputs
    psi_real = np.zeros_like(r)  # Shape (109,)
    psi_imag = np.zeros_like(r)  # Shape (109,)

    for i in range(len(n)):
        try:
            R = radial_wavefunction(n[i], l[i], r).reshape(-1)  # Ensure shape (109,)
            Y_real, Y_imag = angular_wavefunction(l[i], m[i], theta, phi)
            Y_real = Y_real.reshape(-1)  # Ensure shape (109,)
            Y_imag = Y_imag.reshape(-1)  # Ensure shape (109,)

            psi_real += R * Y_real  # Ensure shapes align
            psi_imag += R * Y_imag  # Ensure shapes align
        except ValueError as e:
            print(f"Error with n={n[i]}, l={l[i]}, m={m[i]}: {e}")
        except Exception as e:
            print(f"Unhandled error with n={n[i]}, l={l[i]}, m={m[i]}: {e}")

    return psi_real, psi_imag


class HydrogenWfc():
    def __init__(self, x, y, n, l):
        self.a0 = 1.0
        self.x = x
        self.y = y
        self.n = n
        self.l = l

    def Rnl(self, r, n, l):
        return radial_wavefunction(n, l, r)  # n = n-l-1, alpha = 2*l+1, x = 2*r/(n*a0)

    def normalized_wfc(self, x, y):
        """Return normalized wavefunction."""
        r = np.sqrt(x**2 + y**2)
        wfc = self.Rnl(r, self.n, self.l)


def generate_data(generator, n_range, l_range, m_range, n_samples):
    """
    Generate training data for the wavefunction.
    """
    # Generate spatial samples using Metropolis sampling
    samples = generator.metropolis_sampling(n_samples=n_samples, burn_in=n_samples // 5)
    x, y = samples[:, 0], samples[:, 1]

    # Randomly sample quantum numbers
    n_array = np.random.randint(n_range[0], n_range[1] + 1, size=n_samples)
    l_array = np.random.randint(l_range[0], l_range[1] + 1, size=n_samples)
    m_array = np.random.randint(m_range[0], m_range[1] + 1, size=n_samples)

    # Filter quantum numbers to satisfy constraints
    valid_indices = (l_array < n_array) & (np.abs(m_array) <= l_array)
    n_array, l_array, m_array = n_array[valid_indices], l_array[valid_indices], m_array[valid_indices]
    x, y = x[valid_indices], y[valid_indices]

    # Compute wavefunction values
    psi_real, psi_imag = psi_func(x, y, n_array, l_array, m_array)

    # Stack inputs and labels
    inputs = np.stack((x, y, n_array, l_array, m_array), axis=1)
    psi_values = np.stack((psi_real, psi_imag), axis=1)
    return inputs, psi_values



def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def save_no_terminal(model, path):
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
    quantum_loss = QuantumLoss_XYZ()
    quantum_loss.to(device)
    losses = []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    best_loss = float('inf')
    patience_counter = 0
    patience = 100

    batch = next(iter(data_loader))
    # Should include: "x", "y", "n", "l", "m", "psi_real", "psi_imag"

    for epoch in range(epochs):
        model.train()
        quantum_loss.update_weights(epoch, epochs)
        epoch_loss = 0.0
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch in data_loader:
                x = batch["x"].to(device).requires_grad_(True)
                y = batch["y"].to(device).requires_grad_(True)
                n = batch["n"].to(device)
                l = batch["l"].to(device)
                m = batch["m"].to(device)
                psi_true_real = batch["psi_true_real"].to(device)
                psi_true_imag = batch["psi_true_imag"].to(device)

                r = torch.sqrt(x**2 + y**2).requires_grad_(True)

                # Forward pass
                inputs = torch.stack((x, y), dim=1)  # Only spatial inputs (x, y)
                psi_pred = model(inputs, n, l, m)

                # Forward pass
                optimizer.zero_grad()
                # Compute loss (no need to pass E or V explicitly)
                loss = quantum_loss(psi_pred, x, y, psi_true_real, psi_true_imag)
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
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
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

def main(sample_size, n_range, l_range, m_range, epochs_number):
    set_seed(0)

    def hydrogen_wf_wrapper(pos, n, l, m):
        """
        Wrapper for psi_func to bind x, y with quantum numbers n, l, m.
        Parameters:
            pos: Tuple (x, y) of spatial coordinates.
            n, l, m: Quantum numbers.
        Returns:
            psi_real, psi_imag: Real and imaginary components of the wavefunction.
        """
        x, y = pos[0], pos[1]
        return psi_func(x, y, n, l, m)

    # Data generation
    generator = QMC_gen_v4(
    step_size=0.05,
    wavefunction=lambda pos: hydrogen_wf_wrapper(pos, n=1, l=0, m=0)  # Example defaults
    )
    inputs, psi_values = generate_data(generator, n_range, l_range, m_range, sample_size)
    check_data(inputs, psi_values)

    # Normalize x and y
    scaler = StandardScaler()
    inputs[:, :2] = scaler.fit_transform(inputs[:, :2])  # Normalize only x and y

    # Create dataset and DataLoader
    dataset = QuantumDataset(inputs, psi_values)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

    # Model initialization
    model = WavefunctionNN()
    model.apply(initialize_weights)

    optimizer = optim.Adam(
        model.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6
    )
    quantum_loss = QuantumLoss_XYZ()
    path = model_name

    # Training
    losses = train_model_with_quantum_loss(
        model, quantum_loss, optimizer, data_loader, path, epochs=epochs_number, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Plot loss
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

if __name__ == "__main__":
    main(10000, (1, 4), (0, 3), (-3, 3), 100)
