from .trainer_2 import load_model, save_model
from ..models.base_network_2 import WavefunctionNN
import numpy as np
import torch
import sympy as sp
from sympy import Max, simplify, expand
from collections import defaultdict
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt
from tqdm import tqdm
from numba import jit

model = WavefunctionNN()
state_dict = torch.load('model_4_tanh.pth')
model.load_state_dict(state_dict)
model.eval()
print("Model Loaded Succesfully!")

@jit(nopython=True)
def evaluate_derivatives(r_sym,theta_sym,expression, values):
    """
    Evaluate the derivatives of the wavefunction with respect to r and theta.

    Parameters:
    - expression: sympy.Expr, symbolic representation of the wavefunction.
    - values: dict, dictionary of input values {r, theta, n, l}.

    Returns:
    - wavefunction: float, value of the wavefunction at the input values.
    - dr: float, partial derivative with respect to r.
    - dtheta: float, partial derivative with respect to theta.
    """
    # Compute derivatives
    dr = sp.diff(expression, r_sym)
    dtheta = sp.diff(expression, theta_sym)

    # Evaluate wavefunction and derivatives
    wavefunction = float(expression.subs(values))
    dr_value = float(dr.subs(values))
    dtheta_value = float(dtheta.subs(values))

    return wavefunction, dr_value, dtheta_value

@jit(nopython=True)
def discrete_eigen(model):
     # Parameters
    hbar = 1.0  # Planck's constant (reduced)
    mass = 1.0  # Particle mass
    r_min, r_max = 0.1, 10.0  # Radial range
    theta_min, theta_max = 0, np.pi  # Angular range
    n_r, n_theta = 100, 100  # Number of grid points

    # Create grid
    r = np.linspace(r_min, r_max, n_r)
    theta = np.linspace(theta_min, theta_max, n_theta)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing="ij")
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]


    r_flat = r_grid.flatten()
    theta_flat = theta_grid.flatten()
    n_flat = np.full_like(r_flat, 1)  # Example: n = 1
    l_flat = np.full_like(r_flat, 0)  # Example: l = 0
    inputs = torch.tensor(np.stack((r_flat, theta_flat, n_flat, l_flat), axis=1), dtype=torch.float32)

    # Evaluate the model
    torch.set_grad_enabled(False)
    psi_flat = model(inputs).numpy()
    torch.set_grad_enabled(True)
    
    psi_grid = psi_flat.reshape(r_grid.shape)

    # Radial Laplacian
    laplacian_r = np.diag(-2 * np.ones(n_r)) + np.diag(np.ones(n_r - 1), 1) + np.diag(np.ones(n_r - 1), -1)
    laplacian_r /= dr**2
    laplacian_r += np.diag(2 / r)

    # Angular Laplacian
    laplacian_theta = np.diag(-2 * np.ones(n_theta)) + np.diag(np.ones(n_theta - 1), 1) + np.diag(np.ones(n_theta - 1), -1)
    laplacian_theta /= dtheta**2
    laplacian_total = np.kron(laplacian_r, np.eye(n_theta)) + np.kron(np.eye(n_r), laplacian_theta)

    # Compute Hamiltonian
    V_flat = -1 / r.repeat(n_theta)
    potential_matrix = np.diag(V_flat)
    H = -hbar**2 / (2 * mass) * laplacian_total + potential_matrix

    # Solve eigenvalue problem with tqdm
    eigenvalues, eigenvectors = eigs(H, k=1, which="SM")
    ground_state_energy = np.real(eigenvalues[0])
    ground_state_wavefunction = np.real(eigenvectors[:, 0]).reshape(r_grid.shape)

    print(f"Ground state energy: {ground_state_energy}")

    # Plot wavefunction
    plt.imshow(
        psi_grid,
        extent=[theta_min, theta_max, r_min, r_max],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="Wavefunction Amplitude")
    plt.xlabel("Theta")
    plt.ylabel("Radial Distance r")
    plt.title("Wavefunction \($\psi$(r, $\theta$)\)")
    plt.show()

@jit(nopython=True)
def eigen_xyz(model):
    # Parameters
    hbar = 1.0  # Planck's constant (reduced)
    mass = 1.0  # Particle mass
    r_min, r_max = 0.1, 10.0  # Radial range
    theta_min, theta_max = 0, 2 * np.pi  # Full angular range
    n_r, n_theta = 100, 200  # Number of grid points

    # Create spherical grid
    r = np.linspace(r_min, r_max, n_r)
    theta = np.linspace(theta_min, theta_max, n_theta)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing="ij")

    # Convert to Cartesian coordinates
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    # Flatten grids for model evaluation
    r_flat = r_grid.flatten()
    theta_flat = theta_grid.flatten()
    n_flat = np.full_like(r_flat, 1)  # Example: n = 1
    l_flat = np.full_like(r_flat, 0)  # Example: l = 0
    inputs = torch.tensor(np.stack((r_flat, theta_flat, n_flat, l_flat), axis=1), dtype=torch.float32)

    # Evaluate the model
    torch.set_grad_enabled(False)
    psi_flat = model(inputs).numpy()
    torch.set_grad_enabled(True)

    # Reshape wavefunction back to grid
    psi_grid = psi_flat.reshape(r_grid.shape)

    # Compute probability distribution
    prob_grid = np.abs(psi_grid)**2

    # Discretize Laplacian in Cartesian coordinates
    dx = np.abs(r[1] - r[0]) * np.abs(np.cos(theta[1]) - np.cos(theta[0]))
    dy = np.abs(r[1] - r[0]) * np.abs(np.sin(theta[1]) - np.sin(theta[0]))

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8

    # Discretized Laplacian in Cartesian coordinates
    laplacian_x = np.diag(-2 * np.ones(n_r)) + np.diag(np.ones(n_r - 1), 1) + np.diag(np.ones(n_r - 1), -1)
    laplacian_x /= (dx**2 + epsilon)

    laplacian_y = np.diag(-2 * np.ones(n_theta)) + np.diag(np.ones(n_theta - 1), 1) + np.diag(np.ones(n_theta - 1), -1)
    laplacian_y /= (dy**2 + epsilon)

    # Total Laplacian
    laplacian_total = np.kron(laplacian_x, np.eye(n_theta)) + np.kron(np.eye(n_r), laplacian_y)

    # Potential energy term
    V_flat = -1 / np.sqrt(x_grid.flatten()**2 + y_grid.flatten()**2)
    potential_matrix = np.diag(V_flat)

    # Hamiltonian
    H = -hbar**2 / (2 * mass) * laplacian_total + potential_matrix

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = eigs(H, k=1, which="SM")
    ground_state_energy = np.real(eigenvalues[0])
    ground_state_wavefunction = np.real(eigenvectors[:, 0]).reshape(r_grid.shape)

    print(f"Ground state energy: {ground_state_energy}")

    # Plot probability distribution as a contour plot
    plt.contourf(
        x_grid, y_grid, prob_grid,
        levels=100, cmap="viridis"
    )
    plt.colorbar(label="Probability Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Probability Distribution in Cartesian Coordinates")
    plt.axis("equal")  # Ensure equal aspect ratio for x and y
    plt.show()

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "psi_grid": psi_grid,
        "prob_grid": prob_grid,
        "ground_state_energy": ground_state_energy,
        "ground_state_wavefunction": ground_state_wavefunction
    }

def main():
    eigen_xyz(model)

if __name__ == "__main__":
    main()