"""
Version 5 of the trainer!
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.special import sph_harm
from qmc_generator_5 import QMC_gen_v4
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from base_network_5 import WfcNN
from losses_5 import Losses
from scipy.special import factorial, eval_genlaguerre
import time

# --- Wavefunction Components ---
def radial_wavefunction(n, l, r):
    """
    Compute the radial wavefunction R_{nl} for hydrogen-like atoms.

    Parameters:
        n (np.ndarray): Principal quantum numbers (array).
        l (np.ndarray): Azimuthal quantum numbers (array).
        r (np.ndarray): Radial distances (array).

    Returns:
        np.ndarray: Radial wavefunction values.
    """
    n = np.asarray(n)  # Ensure inputs are arrays
    l = np.asarray(l)
    r = np.asarray(r)

    # Compute rho
    rho = 2 * r / n

    # Compute prefactor
    prefactor = np.sqrt(
        (2 / n)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l))
    )

    # Compute generalized Laguerre polynomials
    laguerre = eval_genlaguerre(n - l - 1, 2 * l, rho)

    # Compute radial wavefunction
    R = prefactor * (rho**l) * np.exp(-rho / 2) * laguerre

    return R


def angular_wavefunction(l, m, theta, phi):
    """
    Compute the spherical harmonics Y_{lm}.

    Parameters:
        l, m (int): Quantum numbers.
        theta, phi (np.ndarray): Angular coordinates.
        theta is the azimuthal 
        phi is the polar

    Returns:
        Y_real, Y_imag (np.ndarray): Real and imaginary parts of spherical harmonics.
    """
    Y = sph_harm(m, l, theta, phi)
    return Y.real, Y.imag

def psi_func(x, y, z, n, l, m):
    """
    Compute the wavefunction (real and imaginary parts).

    Parameters:
        x, y (np.ndarray): Spatial coordinates.
        n, l, m (np.ndarray): Quantum numbers.

    Returns:
        psi_real, psi_imag (np.ndarray): Real and imaginary parts of the wavefunction.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    R = radial_wavefunction(n, l, r)
    Y_real, Y_imag = angular_wavefunction(l, m, theta, phi)
    psi_real = R * Y_real
    psi_imag = R * Y_imag
    return psi_real, psi_imag

# --- Data Sampling ---
def generate_samples(generator, n_range, l_range, m_range, n_samples, n_groups=100):
    """
    Generate valid samples for training with grouped quantum numbers.
    """
    # Placeholder for all sampled data
    sampled_positions = []
    sampled_n = []
    sampled_l = []
    sampled_m = []
    sampled_psi_real = []
    sampled_psi_imag = []

    # Divide samples into groups by quantum numbers
    samples_per_group = n_samples // n_groups

    with tqdm(total=n_groups, desc=f'Metropolis Sampling ({samples_per_group} samples/group)') as pbar:
        for _ in range(n_groups):
            # Dynamically generate one quantum number set for the group
            n_val = np.random.randint(n_range[0], n_range[1] + 1)
            l_val = np.random.randint(0, n_val)  # Ensure l < n
            m_val = np.random.randint(-l_val, l_val + 1)  # Ensure |m| <= l

            # Run Metropolis sampling for this quantum number combination
            group_samples = generator.metropolis_sampling(
                n_samples=samples_per_group, burn_in=generator.burn_in, n=n_val, l=l_val, m=m_val
            )

            for pos in group_samples:
                x, y, z = pos
                psi_real, psi_imag = hydrogen_wavefunction((x, y, z), n_val, l_val, m_val)

                # Append data for this sample
                sampled_positions.append([x, y, z])
                sampled_n.append(n_val)
                sampled_l.append(l_val)
                sampled_m.append(m_val)
                sampled_psi_real.append(psi_real)
                sampled_psi_imag.append(psi_imag)
            pbar.update(1)

    # Convert to numpy arrays
    sampled_positions = np.array(sampled_positions)
    sampled_n = np.array(sampled_n)
    sampled_l = np.array(sampled_l)
    sampled_m = np.array(sampled_m)
    sampled_psi_real = np.array(sampled_psi_real)
    sampled_psi_imag = np.array(sampled_psi_imag)

    # Stack inputs and labels
    inputs = np.column_stack((sampled_positions, sampled_n, sampled_l, sampled_m))
    psi_values = np.column_stack((sampled_psi_real, sampled_psi_imag))

    return inputs, psi_values


def validate_dataset(inputs, psi_values):
    """
    Validate dataset to ensure all quantum numbers are valid.
    """
    n_array, l_array, m_array = inputs[:, 3], inputs[:, 4], inputs[:, 5]
    valid_indices = (n_array >= 1) & (l_array < n_array) & (np.abs(m_array) <= l_array)
    inputs = inputs[valid_indices]
    psi_values = psi_values[valid_indices]
    if len(inputs) == 0:
        raise ValueError("No valid samples found after filtering. Check quantum number generation.")
    return inputs, psi_values

def validate_batch(n, l, m):
    """
    Validate quantum numbers in a batch.
    """
    assert torch.all(n >= 1), f"Invalid n value in batch: {n}"
    assert torch.all(l < n), f"Invalid l value in batch: {l} (l >= n)"
    assert torch.all(torch.abs(m) <= l), f"Invalid m value in batch: {m} (|m| > l)"
    
def hydrogen_wavefunction(pos, n, l, m):
    """
    Wrapper for hydrogen wavefunction with dynamic quantum numbers.
    """
    x, y, z = pos[0], pos[1], pos[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r > 0 else 0
    phi = np.arctan2(y, x)
    R = radial_wavefunction(n, l, r)
    Y_real, Y_imag = angular_wavefunction(l, m, theta, phi)
    psi_real = R * Y_real
    psi_imag = R * Y_imag
    return psi_real, psi_imag

# --- Neural Network Training ---
def initialize_weights_based_on_quantum_numbers(layer, n_max, l_max, m_max):
    """
    Custom weight initialization based on quantum numbers.

    Args:
        layer (nn.Module): Neural network layer to initialize.
        n_max (int): Maximum value of principal quantum number n.
        l_max (int): Maximum value of azimuthal quantum number l.
        m_max (int): Maximum value of magnetic quantum number m.
    """
    if isinstance(layer, nn.Embedding):
        # Initialize embeddings for n, l, and m
        num_embeddings, embedding_dim = layer.weight.size()
        if num_embeddings == n_max + 1:
            # Initialize n embedding (e.g., proportional to 1/n)
            layer.weight.data = torch.randn_like(layer.weight) * (1 / (torch.arange(1, num_embeddings + 1).float()))
        elif num_embeddings == l_max + 1:
            # Initialize l embedding (e.g., proportional to l^2)
            layer.weight.data = torch.randn_like(layer.weight) * (torch.arange(0, num_embeddings).float()**2)
        elif num_embeddings == 2 * m_max + 1:
            # Initialize m embedding (e.g., proportional to m)
            layer.weight.data = torch.randn_like(layer.weight) * torch.arange(-m_max, m_max + 1).float()

    elif isinstance(layer, nn.Linear):
        # Use Xavier initialization for linear layers
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class QuantumDataset(Dataset):
    def __init__(self, data, labels):
        """
        Custom dataset for quantum wavefunction data.

        Parameters:
            data (np.ndarray): Input features [x, y, n, l, m].
            labels (np.ndarray): True wavefunction values [psi_real, psi_imag].
        """
        self.data = data
        self.labels = labels

        # Normalize spatial features
        self.scaler = StandardScaler()
        self.data[:, :3] = self.scaler.fit_transform(self.data[:, :3])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.data[idx, 0], dtype=torch.float32),
            "y": torch.tensor(self.data[idx, 1], dtype=torch.float32),
            "z": torch.tensor(self.data[idx, 2], dtype=torch.float32),
            "n": torch.tensor(self.data[idx, 3], dtype=torch.long),
            "l": torch.tensor(self.data[idx, 4], dtype=torch.long),
            "m": torch.tensor(self.data[idx, 5], dtype=torch.long),
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
    save_path="best_model.pth"
):
    """
    Train the WavefunctionNN model with improved logging and monitoring.

    Args:
        model (nn.Module): WavefunctionNN model.
        dataset (Dataset): QuantumDataset containing input features and labels.
        quantum_loss (Losses): Custom loss function.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        lr (float): Learning rate.
        device (str): Device for training ("cpu" or "cuda").
        save_path (str): Path to save the best model.

    Returns:
        model (nn.Module): Trained model.
        training_losses (list): List of average losses per epoch.
    """
    # Prepare DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    

    # Move model and loss function to the specified device
    model.to(device)
    quantum_loss.to(device)

    # Training metrics
    training_losses = []
    best_loss = float('inf')

    # Training loop
    with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            epoch_losses_dict = {
                "schrodinger_loss": 0.0,
                #"boundary_loss": 0.0,
                #"normalization_loss": 0.0,
                "supervised_loss": 0.0,
            }
            for batch_idx, batch in enumerate(data_loader):
                # Extract batch data and move to device
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                z = batch["z"].to(device)
                n = batch["n"].to(device)
                l = batch["l"].to(device)
                m = batch["m"].to(device)
                psi_true_real = batch["psi_true_real"].to(device)
                psi_true_imag = batch["psi_true_imag"].to(device)
    
                # Validate quantum numbers
                validate_batch(n, l, m)
    
                # Forward pass
                optimizer.zero_grad()
                inputs = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1)  # Combine (x, y, z)
                psi_pred = model(inputs, n, l, m)
    
                # Compute loss
                loss = quantum_loss(psi_pred, x, y, z, psi_true_real, psi_true_imag)
                epoch_loss += loss.item()
    
                # Aggregate individual losses
                #for key in epoch_losses_dict.keys():
                #    epoch_losses_dict[key] += losses_dict[key].item()
    
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            pbar.update(1)

            # Average individual losses for the epoch
            #for key in epoch_losses_dict.keys():
            #    epoch_losses_dict[key] /= len(data_loader)
    
            # Compute average total loss for the epoch
            avg_loss = epoch_loss / len(data_loader)
            training_losses.append(avg_loss)
    
            # Print losses dictionary for the epoch
            #print(f"Epoch {epoch + 1} Loss Breakdown: {epoch_losses_dict}")
    
            # Save the best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_path)
                #print(f"Epoch {epoch + 1}: New best model saved with loss {avg_loss:.6f}")
            scheduler.step(avg_loss)
            #print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {avg_loss:.6f}")
            torch.save(model.state_dict(), save_path)

    return model, training_losses


    

# --- Testing the Data Preparation ---
if __name__ == "__main__":
    start_time = time.time()
    # Generate dataset
    n_range = (1, 4)
    l_range = (0, 3)
    m_range = (-3, 3)
    n_samples = 2500000

    # Metropolis sampler
    generator = QMC_gen_v4(step_size=0.05, wavefunction=hydrogen_wavefunction)
    inputs, psi_values = generate_samples(generator, n_range, l_range,m_range, n_samples)
    inputs, psi_values = validate_dataset(inputs, psi_values)
    dataset = QuantumDataset(inputs, psi_values)

    n_array, l_array, m_array = inputs[:, 3], inputs[:, 4], inputs[:, 5]
    assert np.all(n_array >= 1), "Invalid n detected"
    assert np.all(l_array < n_array), "Invalid l detected (l >= n)"
    assert np.all(np.abs(m_array) <= l_array), "Invalid m detected (|m| > l)"
    print("All quantum numbers are valid!")

    # Initialize model and loss
    model = WfcNN(n_max=4, l_max=3, m_max=3, activation="Tanh")
    quantum_loss = Losses()

    # Train the model
    trained_model, training_losses = train_wavefunction_model(
        model=model,
        dataset=dataset,
        quantum_loss=quantum_loss,
        epochs=100,
        batch_size=128,
        lr=1e-2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path="model_035_tanh.pth"
    )
    
    print(f"Time taken: {time.time()-start_time:.3f} seconds")

