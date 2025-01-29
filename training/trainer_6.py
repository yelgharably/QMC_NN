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
from ..models.base_network_5 import WfcNN
from .losses_6 import Losses
from scipy.special import factorial, eval_genlaguerre
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    laguerre = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)

    # Compute radial wavefunction
    R = prefactor * (rho**l) * np.exp(-rho / 2) * laguerre

    return R


def angular_wavefunction(l, m, theta, phi):
    """
    Compute the spherical harmonics Y_{lm}.

    Parameters:
        l, m (int): Quantum numbers.
        theta, phi (np.ndarray): Angular coordinates.
        theta is the polar 
        phi is the azimuthal

    Returns:
        Y_real, Y_imag (np.ndarray): Real and imaginary parts of spherical harmonics.
    """
    Y = sph_harm(m, l, phi, theta) * (-1)**m
    return Y.real, Y.imag

def psi_func(x, y, z, n, l, m):
    """
    Compute the wavefunction (real and imaginary parts).

    Parameters:
        x, y (np.ndarray): Spatial coordinates.
        n, l, m (np.ndarray): Quantum numbers.

    Returns:
        psi_real, psi_imag (np.ndarray): Real and imagsinary parts of the wavefunction.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    costheta = z / r if r>1e-14 else 1.0
    costheta = np.clip(costheta, -1, 1)
    theta = np.arccos(costheta)
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

def generate_samples_equal_combos(generator, combos, samples_per_combo):
    """
    Generate the same number of samples (samples_per_combo) for each (n, l, m) in combos.
    Returns:
      inputs: np.ndarray of shape [N_total, 6] = (x, y, z, n, l, m)
      psi_values: np.ndarray of shape [N_total, 2] = (psi_real, psi_imag)
    """
    sampled_positions = []
    sampled_n = []
    sampled_l = []
    sampled_m = []
    sampled_psi_real = []
    sampled_psi_imag = []

    # For progress bar
    from tqdm import tqdm
    pbar = tqdm(total=len(combos), desc=f"Metropolis Sampling (equal combos)")

    for (n_val, l_val, m_val) in combos:
        # Run Metropolis sampling for this exact combo
        group_samples = generator.metropolis_sampling(
            n_samples=samples_per_combo,
            burn_in=generator.burn_in,
            n=n_val,
            l=l_val,
            m=m_val
        )

        # For each sampled position, compute "true" wavefunction
        for pos in group_samples:
            x, y, z = pos
            psi_real, psi_imag = hydrogen_wavefunction((x, y, z), n_val, l_val, m_val)

            sampled_positions.append([x, y, z])
            sampled_n.append(n_val)
            sampled_l.append(l_val)
            sampled_m.append(m_val)
            sampled_psi_real.append(psi_real)
            sampled_psi_imag.append(psi_imag)

        pbar.update(1)

    pbar.close()

    # Convert everything to NumPy arrays
    sampled_positions = np.array(sampled_positions)       # shape [N_total, 3]
    sampled_n = np.array(sampled_n)
    sampled_l = np.array(sampled_l)
    sampled_m = np.array(sampled_m)
    sampled_psi_real = np.array(sampled_psi_real)
    sampled_psi_imag = np.array(sampled_psi_imag)

    # Combine into 'inputs' and 'psi_values'
    inputs = np.column_stack((sampled_positions, sampled_n, sampled_l, sampled_m))
    psi_values = np.column_stack((sampled_psi_real, sampled_psi_imag))

    # Optionally shuffle to randomize the order across combos
    # This ensures training samples from all combos are interspersed
    idx = np.arange(len(inputs))
    np.random.shuffle(idx)
    inputs = inputs[idx]
    psi_values = psi_values[idx]

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
    costheta = z / r if r>1e-14 else 1.0
    costheta = np.clip(costheta, -1, 1)
    theta = np.arccos(costheta)
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
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{percentage:3.0f}% | Loss: {postfix} | ETA: {remaining}s]"
        ) as pbar:
            progress_accum = 0.0
            last_printed   = 0
            for batch_idx, batch in enumerate(data_loader):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                z = batch["z"].to(device)
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
                coords = torch.stack([x, y, z], dim=1)
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
    # Generate dataset
    n_range = (1, 4)
    l_range = (0, 3)
    m_range = (-3, 3)
    n_samples = 25000
    combos = generate_nlm_combos(n_max=4)
    samples_per_combo = n_samples // len(combos)

    # Metropolis sampler
    generator = QMC_gen_v4(step_size=0.15, wavefunction=hydrogen_wavefunction)
    inputs, psi_values = generate_samples_equal_combos(generator, combos, samples_per_combo)
    inputs, psi_values = validate_dataset(inputs, psi_values)
    dataset = QuantumDataset(inputs, psi_values)

    n_array, l_array, m_array = inputs[:, 3], inputs[:, 4], inputs[:, 5]
    assert np.all(n_array >= 1), "Invalid n detected"
    assert np.all(l_array < n_array), "Invalid l detected (l >= n)"
    assert np.all(np.abs(m_array) <= l_array), "Invalid m detected (|m| > l)"
    print("All quantum numbers are valid!")

    # Initialize model and loss
    model = WfcNN(n_max=4, l_max=3, m_max=3, activation="Tanh")
    quantum_loss = Losses(model)

    # Train the model
    trained_model, training_losses = train_wavefunction_model(
        model=model,
        dataset=dataset,
        quantum_loss=quantum_loss,
        epochs=10,
        batch_size=64,
        lr=1e-2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path="model_048_tanh.pth"
    )
    
    print(f"Time taken: {time.time()-start_time:.3f} seconds")

