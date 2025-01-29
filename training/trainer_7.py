"""
Version 6 of the trainer!
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.special import sph_harm
from ..qmc.qmc_generator_5 import QMC_gen_v4
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from ..models.base_network_6 import WfcNN
from .losses_7 import Losses
from scipy.special import factorial, eval_genlaguerre
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
eps = 1e-8

def generate_nlm_combos(n_max=4):
    combos = []
    for n in range(1, n_max + 1):
        for l in range(n):
            for m in range(-l, l + 1):
                combos.append((n, l, m))
    return combos

def initialize_weights(module):
    """
    Applies appropriate initialization based on the type of module.
    """
    if isinstance(module, nn.Linear):
        # Xavier Uniform Initialization for Linear Layers
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Initialize Embedding Layers
        nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Small random values for embeddings

# --- Wavefunction Components ---
def radial_wavefunction(n, l, r):
    rho = 2 * r / n
    prefactor = np.sqrt((2 / n)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    laguerre = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    R = prefactor * (rho**l) * np.exp(-rho / 2) * laguerre
    return R

def angular_wavefunction(l, m, phi):
    Y = sph_harm(m, l, phi, np.pi/2) * (-1)**m
    return Y.real, Y.imag

def psi_func(x, y, n, l, m):
    r = np.sqrt(x**2 + y**2) + eps
    phi = np.arctan2(y, x)
    R = radial_wavefunction(n, l, r)
    Y_real, Y_imag = angular_wavefunction(l, m, phi)
    psi_real = R * Y_real
    psi_imag = R * Y_imag
    return psi_real, psi_imag


# --- Data Generation ---
def generate_uniform_grid(grid_size, max_radius):
    linspace = np.linspace(-max_radius, max_radius, grid_size)
    x, y = np.meshgrid(linspace, linspace)
    return x.flatten(), y.flatten()

def generate_samples_uniform_grid(n_max, grid_size, max_radius):
    x, y = generate_uniform_grid(grid_size, max_radius)
    inputs = []
    psi_values = []

    combos = generate_nlm_combos(n_max)
    total_points = len(combos) * len(x)
    with tqdm(total=total_points, desc="Generating Samples") as pbar:
        for (n, l, m) in combos:
            for xi, yi in zip(x, y):
                psi_real, psi_imag = psi_func(xi, yi, n, l, m)
                inputs.append([xi, yi, n, l, m])
                psi_values.append([psi_real, psi_imag])
                pbar.update(1)

    return np.array(inputs), np.array(psi_values)


def validate_batch(n, l, m):
    """
    Validate quantum numbers in a batch.
    """
    assert torch.all(n >= 1), f"Invalid n value in batch: {n}"
    assert torch.all(l < n), f"Invalid l value in batch: {l} (l >= n)"
    assert torch.all(torch.abs(m) <= l), f"Invalid m value in batch: {m} (|m| > l)"

class QuantumDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.data[idx, 0], dtype=torch.float32),
            "y": torch.tensor(self.data[idx, 1], dtype=torch.float32),
            "n": torch.tensor(self.data[idx, 2], dtype=torch.long),
            "l": torch.tensor(self.data[idx, 3], dtype=torch.long),
            "m": torch.tensor(self.data[idx, 4], dtype=torch.long),
            "psi_true_real": torch.tensor(self.labels[idx, 0], dtype=torch.float32),
            "psi_true_imag": torch.tensor(self.labels[idx, 1], dtype=torch.float32),
        }


def train_wavefunction_model(
    model, 
    dataset, 
    quantum_loss, 
    epochs=100, 
    batch_size=128, 
    lr=1e-3, 
    device="cpu", 
    save_path="best_model.pth",
    pdf_vals=None
):
    """
    Trains the WfcNN model. We assume quantum_loss is the revised Losses class
    that re-runs model inside it.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    optimizer = Adam(model.parameters(), lr=lr)
    num_batches = len(data_loader)
    batch_to_percent = 100.0 / num_batches

    scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',          # we want to minimize the loss
                    factor=0.5,
                    patience=5,
                    threshold=1e-3,      # how much improvement is "significant"
                    cooldown=0,
                    min_lr=1e-6,
                    verbose=True         # print LR changes
                )

    model.to(device)

    quantum_loss.to(device)

    training_losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        with tqdm(
            total=100, 
            desc=f"Epoch {epoch+1}", 
            unit="%", 
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{percentage:3.0f}% | Loss: | ETA: {remaining}s]"
        ) as pbar:
            progress_accum = 0.0
            last_printed   = 0
            for batch_idx, batch in enumerate(data_loader):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                n = batch["n"].to(device)
                l = batch["l"].to(device)
                m = batch["m"].to(device)
                psi_true_real = batch["psi_true_real"].to(device)
                psi_true_imag = batch["psi_true_imag"].to(device)
                progress_accum += batch_to_percent
                next_int = int(progress_accum)

                validate_batch(n, l, m)

                epoch_losses_accum = {
                "total_loss": 0.0,
                "schrodinger_loss": 0.0,
                "supervised_loss": 0.0,
                "normalization_loss": 0.0,
                }

                # Construct coords = [batch_size, 3]
                coords = torch.stack([x, y], dim=1)  # Only x and y
                # print("coords.shape =", coords.shape)
                # print("n.shape =", n.shape, "l.shape =", l.shape, "m.shape =", m.shape)


                # Zero grads
                optimizer.zero_grad()

                # Optionally, if you want MC normalization, you need pdf_vals[batch_idx] 
                # or something that lines up with coords. For a big dataset, 
                # you'd store pdf_vals in dataset, too.
                # In this minimal snippet, we pass None for pdf_vals.
                total_loss, loss_dict = quantum_loss(
                    coords=coords,
                    n=n, l=l, m=m,
                    psi_true_real=psi_true_real,
                    psi_true_imag=psi_true_imag,
                    pdf_vals=None  # or your array if implementing normalization
                )

                epoch_losses_accum["total_loss"]          += total_loss.item()
                epoch_losses_accum["schrodinger_loss"]    += loss_dict["schrodinger_loss"].item()
                epoch_losses_accum["supervised_loss"]     += loss_dict["supervised_loss"].item()
                epoch_losses_accum["normalization_loss"]  += loss_dict["normalization_loss"].item()

                total_loss.backward()
                optimizer.step()

                epoch_loss_sum += total_loss.item()

                avg_loss_so_far = epoch_loss_sum / (batch_idx + 1)
                if next_int > last_printed:
                    # We only update by the difference (so we don't jump past multiple integers)
                    pbar.update(next_int - last_printed)
                    pbar.set_postfix(loss=f"{avg_loss_so_far:.4f}")
    
                    # Advance our "last_printed" marker
                    last_printed = next_int

            avg_loss = epoch_loss_sum / len(data_loader)
            training_losses.append(avg_loss)
            if last_printed < 100:
                pbar.update(100 - last_printed)

            # Save model if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_path)
            scheduler.step(avg_loss)

            num_batches = len(data_loader)
            for key in epoch_losses_accum.keys():
                epoch_losses_accum[key] /= num_batches

            # Record total loss for plotting or analysis
            training_losses.append(epoch_losses_accum["total_loss"])
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            print(f"  total_loss:          {epoch_losses_accum['total_loss']:.6f}")
            print(f"  schrodinger_loss:    {epoch_losses_accum['schrodinger_loss']:.6f}")
            print(f"  supervised_loss:     {epoch_losses_accum['supervised_loss']:.6f}")
            print(f"  normalization_loss:  {epoch_losses_accum['normalization_loss']:.6f}")

    return model, training_losses
    

# --- Testing the Data Preparation ---
if __name__ == "__main__":
    start_time = time.time()
    n_max = 2
    grid_size = 150
    max_radius = 10

    inputs, psi_values = generate_samples_uniform_grid(n_max, grid_size, max_radius)
    dataset = QuantumDataset(inputs, psi_values)

    # Initialize model and loss (model definition required)
    model = WfcNN(n_max=4, l_max=3, m_max=3, activation="Tanh")
    quantum_loss = Losses(model)

    # Train the model (training function required)
    trained_model, training_losses = train_wavefunction_model(
        model=model,
        dataset=dataset,
        quantum_loss=quantum_loss,
        epochs=20,
        batch_size=64,
        lr=1e-2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path="model_053_tanh.pth"
    )

    print(f"Time taken: {time.time()-start_time:.3f} seconds")
