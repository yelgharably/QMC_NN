# eval.py

from .trainer_2 import load_model, save_model
from ..models.base_network_2 import WavefunctionNN
import numpy as np
import torch
from scipy.sparse.linalg import eigs
from scipy.sparse import kron,eye,diags
from matplotlib import pyplot as plt
from tqdm import tqdm
from numba import njit
import sys

# Load the model
model = WavefunctionNN()
model_num = sys.argv[1]
state_dict = torch.load(f'model_{model_num}_tanh.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
print("Model Loaded Successfully!")

# Numba-accelerated function to construct Laplacian matrices in Cartesian coordinates
def construct_cartesian_laplacian_sparse(x, dx, y, dy):
    n_x = len(x)
    n_y = len(y)

    # 1D Laplacian for x
    diagonals_x = [-2.0 / (dx ** 2), 1.0 / (dx ** 2), 1.0 / (dx ** 2)]
    laplacian_x = diags(diagonals_x, [0, -1, 1], shape=(n_x, n_x))

    # 1D Laplacian for y
    diagonals_y = [-2.0 / (dy ** 2), 1.0 / (dy ** 2), 1.0 / (dy ** 2)]
    laplacian_y = diags(diagonals_y, [0, -1, 1], shape=(n_y, n_y))

    # Combine using Kronecker product
    laplacian_total = kron(laplacian_x, eye(n_y)) + kron(eye(n_x), laplacian_y)

    return laplacian_total

# Function to compute ground state energy and wavefunction
def compute_ground_state(H):
    eigenvalues, eigenvectors = eigs(H, k=1, which="SM")
    ground_state_energy = np.real(eigenvalues)
    ground_state_wavefunction = np.real(eigenvectors[:, 0])
    return ground_state_energy, ground_state_wavefunction

# Function to plot wavefunction in Cartesian coordinates
def plot_wavefunction_cartesian(x_grid, y_grid, wavefunction_grid, title):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x_grid, y_grid, wavefunction_grid, shading='auto', cmap='viridis')
    plt.colorbar(label="Wavefunction Amplitude")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis('equal')
    plt.show()

# Function to compute eigenvalues and wavefunctions in Cartesian coordinates
def eigen_xyz(model):
    # Parameters
    hbar = 1.0
    mass = 1.0
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0, 3.0
    n_x, n_y = 200, 200

    # Create Cartesian grid
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Flatten grids for model evaluation
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    n_flat = np.full_like(x_flat, 1)  # Example: n = 1
    l_flat = np.full_like(x_flat, 0)  # Example: l = 0
    inputs = np.stack((x_flat, y_flat, n_flat, l_flat), axis=1)

    # Convert to PyTorch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)

    # Evaluate the model
    with torch.no_grad():
        psi_flat = model(inputs_tensor).cpu().numpy()

    # Reshape to grid
    psi_grid = psi_flat.reshape(x_grid.shape)

    psi_squared_grid = np.abs(psi_grid)**2
    normalization_factor = np.sum(psi_squared_grid * dx * dy)
    psi_squared_grid /= normalization_factor

    # Construct Laplacian matrices
    laplacian_total = construct_cartesian_laplacian_sparse(x, dx, y, dy)
    print("Laplacian shape:", laplacian_total.shape)
    print("Laplacian sparsity:", laplacian_total.nnz / (laplacian_total.shape[0]**2))

    # Compute potential energy matrix
    V_flat = -1 / np.sqrt(x_flat**2 + y_flat**2 + 1e-8)
    print("Potential Energy Min:", V_flat.min(), "Max:", V_flat.max())
    V_matrix = np.diag(V_flat)

    # Construct Hamiltonian matrix
    H = -hbar**2 / (2 * mass) * laplacian_total + V_matrix

    # Compute ground state energy and wavefunction
    ground_state_energy, ground_state_wavefunction = compute_ground_state(H)
    ground_state_wavefunction_grid = ground_state_wavefunction.reshape(x_grid.shape)
    print(ground_state_energy)

    integral = np.sum(psi_squared_grid * dx * dy)
    print("Wavefunction normalization check:", integral)


    plt.figure(figsize=(8, 6))
    plt.contourf(x_grid, y_grid, psi_squared_grid, levels=50, cmap='viridis')
    plt.colorbar(label="Probability Density ( |$\psi$|$^2$)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Probability Density (|$\psi$|$^2$) in Cartesian Coordinates")
    plt.axis('equal')  # Ensure equal scaling for x and y axes
    plt.show()

    plt.spy(laplacian_total, markersize=1)
    plt.title("Laplacian Sparsity Pattern")
    plt.show()
    plt.spy(H, markersize=1)
    plt.title("Hamiltonian Sparsity Pattern")
    plt.show()


    # Plot wavefunctions
    #plot_wavefunction_cartesian(x_grid, y_grid, psi_grid, "Predicted Wavefunction")
    plot_wavefunction_cartesian(x_grid, y_grid, ground_state_wavefunction_grid, "Ground State Wavefunction")

    return ground_state_energy, ground_state_wavefunction_grid

def main():
    eigen_xyz(model)

if __name__ == "__main__":
    main()
